import AutoAnimMacCore
import SwiftUI

struct ReviewWorkspaceView: View {
    let job: JobSummary
    @ObservedObject var model: AppModel
    let endpoint: LoopbackEndpoint
    let token: String

    @State private var workspace: ReviewWorkspaceState?
    @State private var selectedComparisonJobID = ""
    @State private var frameIndex = 0
    @State private var commandsA: [ReviewBridgeEnvelope] = []
    @State private var commandsB: [ReviewBridgeEnvelope] = []
    @State private var errorMessage: String?
    @State private var bridgeBlockedA = false
    @State private var bridgeBlockedB = false
    @State private var timelineIsEditing = false
    @State private var layerStateA = ViewerLayerVisibility()
    @State private var layerStateB = ViewerLayerVisibility()
    @State private var reloadGenerationA = 0
    @State private var reloadGenerationB = 0

    var body: some View {
        Group {
            if let bundle = model.reviewBundles[job.jobID], workspace != nil {
                review(bundle)
            } else if model.loadingReviewJobIDs.contains(job.jobID) {
                ProgressView("Verifying sealed review evidence…")
            } else {
                ContentUnavailableView(
                    "Native review unavailable",
                    systemImage: "exclamationmark.shield",
                    description: Text(
                        model.reviewBundleErrors[job.jobID]
                            ?? "This video job does not expose a verified ReviewBundle."
                    )
                )
            }
        }
        .task(id: job.jobID) {
            await model.discoverComparisonCandidates(for: job.jobID)
            initializeWorkspace()
        }
        .onChange(of: model.reviewBundles[job.jobID]) { _, _ in
            if workspace == nil { initializeWorkspace() }
        }
        .onChange(of: selectedComparisonJobID) { _, _ in
            loadSelectedComparison()
        }
    }

    private func review(_ bundle: ReviewBundle) -> some View {
        VStack(spacing: 0) {
            reviewToolbar(bundle)
            Divider()
            viewerArea(bundle)
            Divider()
            timeline(bundle)
        }
        .overlay(alignment: .bottomTrailing) {
            if let errorMessage {
                VStack(alignment: .leading, spacing: 6) {
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundStyle(.red)
                    Button("Reload review viewers") { reloadViewers() }
                        .controlSize(.small)
                }
                    .padding(9)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                    .padding(12)
            }
        }
    }

    private func reviewToolbar(_ bundle: ReviewBundle) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 10) {
                Label("Verified artifact ledger", systemImage: "shield.lefthalf.filled")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Text(String(bundle.bundleSHA256.prefix(12)))
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                Spacer()
                Picker("Compare", selection: $selectedComparisonJobID) {
                    Text("No comparison").tag("")
                    ForEach(model.compatibleReviewJobs(for: job.jobID)) { candidate in
                        Text("\(candidate.input.name) · \(candidate.jobID.suffix(8))")
                            .tag(candidate.jobID)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 300)
                .disabled(!commandsA.isEmpty || !commandsB.isEmpty)
                if workspace?.reviewB != nil {
                    ControlGroup {
                        Button("A") { selectSlot(.a) }
                            .disabled(workspace?.activeSlot == .a)
                        Button("B") { selectSlot(.b) }
                            .disabled(
                                workspace?.isReviewBEnabled != true
                                    || workspace?.activeSlot == .b
                            )
                    }
                    .controlSize(.small)
                }
            }

            Label(
                "UNAPPROVED CANDIDATE · NOT PRODUCTION VALIDATED · REPORTED MOTION IS NOT INDEPENDENT APPROVAL",
                systemImage: "exclamationmark.triangle.fill"
            )
            .font(.caption2.monospaced().weight(.bold))
            .foregroundStyle(.orange)

            HStack(spacing: 6) {
                ForEach(bundle.layers, id: \.layerID.rawValue) { layer in
                    Label(
                        layerLabel(layer.layerID),
                            systemImage: layer.changesMotionReported
                            ? "waveform.path"
                            : (layer.availability == "available" ? "circle" : "minus.circle")
                    )
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 4)
                    .background(.quaternary, in: Capsule())
                    .help(
                        "\(layer.availability) · reported authority: \(layer.motionAuthority) · "
                            + "production authority: \(layer.productionMotionAuthority) · "
                            + "approval: \(layer.approvalStatus)"
                    )
                }
            }

            HStack(spacing: 6) {
                ForEach(ReviewViewerLayerID.allCases, id: \.rawValue) { layer in
                    Button {
                        toggleViewerLayer(layer)
                    } label: {
                        Label(
                            viewerLayerLabel(layer),
                            systemImage: layerIsVisible(layer) ? "eye.fill" : "eye.slash"
                        )
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(reviewControlsBlocked)
                }
                Spacer()
                Button("Home") { selectCamera(.home) }
                    .controlSize(.small)
                    .disabled(reviewControlsBlocked)
                Button("Front") { selectCamera(.front) }
                    .controlSize(.small)
                    .disabled(reviewControlsBlocked)
            }
        }
        .padding(10)
        .background(.bar)
    }

    @ViewBuilder
    private func viewerArea(_ bundle: ReviewBundle) -> some View {
        if let comparison = workspace?.reviewB {
            ZStack {
                viewerPane(bundle: bundle, slot: .a, commands: commandsA)
                    .opacity(workspace?.activeSlot == .a ? 1 : 0)
                    .allowsHitTesting(workspace?.activeSlot == .a)
                viewerPane(bundle: comparison, slot: .b, commands: commandsB)
                    .opacity(workspace?.activeSlot == .b ? 1 : 0)
                    .allowsHitTesting(workspace?.activeSlot == .b)
            }
        } else {
            viewerPane(bundle: bundle, slot: .a, commands: commandsA)
        }
    }

    private func viewerPane(
        bundle: ReviewBundle,
        slot: ReviewWorkspaceSlot,
        commands: [ReviewBridgeEnvelope]
    ) -> some View {
        VStack(spacing: 0) {
            HStack {
                Text(slot == .a ? "A · SELECTED" : "B · COMPARISON")
                    .font(.caption.monospaced().weight(.bold))
                Spacer()
                if workspace?.activeSlot == slot {
                    Text("ACTIVE")
                        .font(.caption2.monospaced().weight(.bold))
                        .foregroundStyle(.blue)
                }
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 6)
            .background(.bar)

            AuthenticatedViewer(
                endpoint: endpoint,
                token: token,
                jobID: bundle.sourceManifest.jobID,
                reviewBridge: .init(
                    bundle: bundle,
                    commands: commands,
                    onDocumentReset: {
                        resetViewerDocument(slot)
                    },
                    onCommandAccepted: { commandAccepted($0, for: slot) },
                    onReceive: { receive($0, from: slot) },
                    onError: { blockBridge($0, for: slot) }
                )
            )
            .id(
                "\(bundle.bundleSHA256)-\(slot.rawValue)-"
                    + "\(slot == .a ? reloadGenerationA : reloadGenerationB)"
            )
            .onTapGesture { selectSlot(slot) }
        }
    }

    private func timeline(_ bundle: ReviewBundle) -> some View {
        VStack(spacing: 7) {
            HStack(spacing: 10) {
                Button { step(-1) } label: {
                    Image(systemName: "backward.frame.fill")
                }
                .disabled(frameIndex == 0 || reviewControlsBlocked)

                Slider(
                    value: Binding(
                        get: { Double(frameIndex) },
                        set: { frameIndex = Int($0.rounded()) }
                    ),
                    in: 0...Double(max(0, bundle.clock.frameCount - 1)),
                    step: 1,
                    onEditingChanged: { editing in
                        timelineIsEditing = editing
                        if !editing { seekCurrentFrame(operation: .seek) }
                    }
                )
                .disabled(reviewControlsBlocked)

                Button { step(1) } label: {
                    Image(systemName: "forward.frame.fill")
                }
                .disabled(
                    frameIndex >= bundle.clock.frameCount - 1
                        || reviewControlsBlocked
                )
            }

            HStack(spacing: 12) {
                Text("FRAME \(frameIndex + 1)/\(bundle.clock.frameCount)")
                Text("PTS \(bundle.clock.sourcePTS[frameIndex])")
                Text(displayTime(bundle))
                Text("TB \(bundle.clock.timeBase.numerator)/\(bundle.clock.timeBase.denominator)")
                Spacer()
                Label(syncLabel, systemImage: syncSymbol)
                    .foregroundStyle(syncColor)
                if workspace?.reviewB != nil {
                    Text("CAMERA ORBIT UNVERIFIED")
                        .foregroundStyle(.orange)
                }
            }
            .font(.caption.monospaced())
        }
        .padding(10)
        .background(.bar)
    }

    private func initializeWorkspace() {
        guard workspace == nil else { return }
        guard let bundle = model.reviewBundles[job.jobID] else { return }
        do {
            workspace = try ReviewWorkspaceState(reviewA: bundle)
            frameIndex = 0
            commandsA = []
            commandsB = []
            errorMessage = nil
            bridgeBlockedA = false
            bridgeBlockedB = false
            selectedComparisonJobID = ""
            layerStateA = ViewerLayerVisibility()
            layerStateB = ViewerLayerVisibility()
            reloadGenerationA = 0
            reloadGenerationB = 0
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func loadSelectedComparison() {
        guard var current = workspace else { return }
        do {
            commandsB = []
            bridgeBlockedB = false
            if selectedComparisonJobID.isEmpty {
                current.unloadReviewB()
            } else if let candidate = model.reviewBundles[selectedComparisonJobID] {
                try current.loadReviewB(candidate)
            }
            workspace = current
            if !bridgeBlockedA { errorMessage = nil }
        } catch {
            selectedComparisonJobID = ""
            errorMessage = error.localizedDescription
        }
    }

    private func receive(
        _ envelope: ReviewBridgeEnvelope,
        from slot: ReviewWorkspaceSlot
    ) {
        guard var current = workspace else { return }
        do {
            let result = try current.receive(envelope, from: slot)
            workspace = current
            if case .layerChanged(let layer) = envelope.payload {
                setLayer(layer.layerID, visible: layer.visible, for: slot)
            }
            if case .viewerError(let error) = result {
                setBridgeBlocked(true, for: slot)
                errorMessage = "Viewer \(slot.rawValue.uppercased()) · "
                    + "\(error.code.rawValue) · \(error.detail)"
                return
            }
            continueReviewHandshake(for: slot)
        } catch {
            setBridgeBlocked(true, for: slot)
            errorMessage = error.localizedDescription
        }
    }

    private func continueReviewHandshake(for slot: ReviewWorkspaceSlot) {
        guard !isBridgeBlocked(slot) else { return }
        guard var proposed = workspace else { return }
        do {
            let viewer = slot == .a ? proposed.viewerA : proposed.viewerB
            let slotExists = slot == .a || proposed.reviewB != nil
            var proposedCommand: ReviewBridgeEnvelope?
            if slotExists,
               viewer.frameCount != nil,
               viewer.revisionReady,
               let missing = nextMissingLayer(in: viewer) {
                proposedCommand = try proposed.requestLayer(
                    missing,
                    visible: desiredLayerVisibility(missing, for: slot, in: proposed),
                    for: slot
                ).envelope
            } else if slotExists,
                      viewer.frameCount != nil,
                      viewer.revisionReady,
                      viewer.hasCompleteAcknowledgedLayerState,
                      viewer.acknowledgedSelection == nil {
                let other = slot == .a ? proposed.viewerB : proposed.viewerA
                proposedCommand = try proposed.requestSelection(
                    other.acknowledgedSelection ?? .region(.none),
                    for: slot
                ).envelope
            }

            if let proposedCommand {
                try commit(
                    [ReviewWorkspaceCommand(slot: slot, envelope: proposedCommand)],
                    state: proposed
                )
                return
            }

            let renderStateReady = proposed.viewerA.hasCompleteAcknowledgedLayerState
                && proposed.viewerA.acknowledgedSelection != nil
                && (proposed.reviewB == nil || proposed.crossBundleRenderStateMatches)
            if proposed.targetFrame == nil,
               renderStateReady,
               proposed.viewerA.supportsExactCursor,
               (proposed.reviewB == nil || proposed.viewerB.supportsExactCursor) {
                workspace = proposed
                seekCurrentFrame(operation: .pause)
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func nextMissingLayer(
        in viewer: ReviewWorkspaceViewerState
    ) -> ReviewViewerLayerID? {
        ReviewViewerLayerID.allCases.first { candidate in
            !viewer.acknowledgedLayers.contains { $0.layerID == candidate }
        }
    }

    private func desiredLayerVisibility(
        _ layer: ReviewViewerLayerID,
        for slot: ReviewWorkspaceSlot,
        in state: ReviewWorkspaceState
    ) -> Bool {
        let other = slot == .a ? state.viewerB : state.viewerA
        if let acknowledged = other.acknowledgedLayers.first(
            where: { $0.layerID == layer }
        ) {
            return acknowledged.visible
        }
        return (slot == .a ? layerStateA : layerStateB).isVisible(layer)
    }

    private func seekCurrentFrame(operation: ReviewCursorOperation) {
        guard var current = workspace else { return }
        do {
            if current.reviewB != nil {
                let commands = try current.requestComparisonFrame(
                    frameIndex,
                    operation: operation
                )
                try commit(commands, state: current)
            } else {
                let command = try current.requestFrame(
                    frameIndex,
                    for: .a,
                    operation: operation
                )
                try commit([command], state: current)
            }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func step(_ delta: Int) {
        guard let bundle = workspace?.reviewA else { return }
        frameIndex = min(max(frameIndex + delta, 0), bundle.clock.frameCount - 1)
        seekCurrentFrame(operation: .step)
    }

    private func selectSlot(_ slot: ReviewWorkspaceSlot) {
        guard var current = workspace else { return }
        do {
            try current.select(slot)
            workspace = current
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func reloadViewers() {
        guard var current = workspace else { return }
        current.resetViewerDocument(for: .a)
        reloadGenerationA += 1
        if current.reviewB != nil {
            current.resetViewerDocument(for: .b)
            reloadGenerationB += 1
        }
        workspace = current
        commandsA = []
        commandsB = []
        bridgeBlockedA = false
        bridgeBlockedB = false
        errorMessage = nil
    }

    private func resetViewerDocument(_ slot: ReviewWorkspaceSlot) {
        guard var current = workspace else { return }
        current.resetViewerDocument(for: slot)
        workspace = current
        if slot == .a {
            commandsA = []
            bridgeBlockedA = false
        } else {
            commandsB = []
            bridgeBlockedB = false
        }
    }

    private func toggleViewerLayer(_ layer: ReviewViewerLayerID) {
        guard var proposed = workspace else { return }
        let visible = !layerIsVisible(layer)
        do {
            let proposedA = try proposed.requestLayer(
                layer,
                visible: visible,
                for: .a
            ).envelope
            let proposedB: ReviewBridgeEnvelope?
            if proposed.reviewB != nil {
                proposedB = try proposed.requestLayer(
                    layer,
                    visible: visible,
                    for: .b
                ).envelope
            } else {
                proposedB = nil
            }
            var commands = [
                ReviewWorkspaceCommand(slot: .a, envelope: proposedA)
            ]
            if let proposedB {
                commands.append(
                    ReviewWorkspaceCommand(slot: .b, envelope: proposedB)
                )
            }
            try commit(commands, state: proposed)
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func selectCamera(_ preset: ReviewCameraPreset) {
        guard var proposed = workspace else { return }
        do {
            let proposedA = try proposed.requestSelection(
                .cameraPreset(preset),
                for: .a
            ).envelope
            let proposedB: ReviewBridgeEnvelope?
            if proposed.reviewB != nil {
                proposedB = try proposed.requestSelection(
                    .cameraPreset(preset),
                    for: .b
                ).envelope
            } else {
                proposedB = nil
            }
            var commands = [
                ReviewWorkspaceCommand(slot: .a, envelope: proposedA)
            ]
            if let proposedB {
                commands.append(
                    ReviewWorkspaceCommand(slot: .b, envelope: proposedB)
                )
            }
            try commit(commands, state: proposed)
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func layerIsVisible(_ layer: ReviewViewerLayerID) -> Bool {
        let state = workspace?.activeSlot == .b ? layerStateB : layerStateA
        return state.isVisible(layer)
    }

    private func setLayer(
        _ layer: ReviewViewerLayerID,
        visible: Bool,
        for slot: ReviewWorkspaceSlot
    ) {
        if slot == .a {
            layerStateA.set(layer, visible: visible)
        } else {
            layerStateB.set(layer, visible: visible)
        }
    }

    private func commit(
        _ newCommands: [ReviewWorkspaceCommand],
        state: ReviewWorkspaceState
    ) throws {
        var proposedA = commandsA
        var proposedB = commandsB
        for command in newCommands {
            let slot = command.slot
            guard !isBridgeBlocked(slot) else {
                throw ReviewWorkspaceViewError.bridgeBlocked(slot)
            }
            var proposed = slot == .a ? proposedA : proposedB
            guard proposed.count < 256 else {
                throw ReviewWorkspaceViewError.commandQueueLimit(slot)
            }
            let expectedJobID = slot == .a
                ? state.reviewA.sourceManifest.jobID
                : state.reviewB?.sourceManifest.jobID
            guard command.envelope.jobID == expectedJobID else {
                throw ReviewWorkspaceViewError.commandJobMismatch(slot)
            }
            if let last = proposed.last,
               (last.jobID != command.envelope.jobID
                   || last.sequence >= command.envelope.sequence) {
                throw ReviewWorkspaceViewError.commandSequence(slot)
            }
            proposed.append(command.envelope)
            if slot == .a {
                proposedA = proposed
            } else {
                proposedB = proposed
            }
        }
        workspace = state
        commandsA = proposedA
        commandsB = proposedB
    }

    private func commandAccepted(
        _ command: ReviewBridgeEnvelope,
        for slot: ReviewWorkspaceSlot
    ) {
        var commands = slot == .a ? commandsA : commandsB
        guard let first = commands.first else { return }
        guard first.jobID == command.jobID,
              first.sequence == command.sequence
        else {
            setBridgeBlocked(true, for: slot)
            errorMessage = ReviewWorkspaceViewError.commandAcceptanceOrder(slot)
                .localizedDescription
            return
        }
        commands.removeFirst()
        if slot == .a {
            commandsA = commands
        } else {
            commandsB = commands
        }
    }

    private func blockBridge(
        _ error: AuthenticatedViewer.ReviewBridgeError,
        for slot: ReviewWorkspaceSlot
    ) {
        setBridgeBlocked(true, for: slot)
        errorMessage = "Viewer \(slot.rawValue.uppercased()) · "
            + error.localizedDescription
    }

    private func isBridgeBlocked(_ slot: ReviewWorkspaceSlot) -> Bool {
        slot == .a ? bridgeBlockedA : bridgeBlockedB
    }

    private func setBridgeBlocked(
        _ blocked: Bool,
        for slot: ReviewWorkspaceSlot
    ) {
        if slot == .a {
            bridgeBlockedA = blocked
        } else {
            bridgeBlockedB = blocked
        }
    }

    private var reviewControlsBlocked: Bool {
        bridgeBlockedA || (workspace?.reviewB != nil && bridgeBlockedB)
    }

    private func displayTime(_ bundle: ReviewBundle) -> String {
        let delta = bundle.clock.sourcePTS[frameIndex] - bundle.clock.firstSourcePTS
        let seconds = Double(delta * bundle.clock.timeBase.numerator)
            / Double(bundle.clock.timeBase.denominator)
        return String(format: "DISPLAY %.3f s", seconds)
    }

    private var syncLabel: String {
        guard let workspace else { return "NOT READY" }
        guard !timelineIsEditing,
              workspace.targetFrame?.frameIndex == frameIndex
        else { return "TARGET NOT APPLIED" }
        if workspace.reviewB != nil {
            return workspace.isReviewBEnabled
                ? "CURSOR/PTS + DECLARED LAYERS MATCH"
                : "A/B PENDING"
        }
        switch workspace.reviewAFrameSynchronization {
        case .exact: return "EXACT"
        case .pending: return "PENDING"
        case .idle: return "READY FOR SEEK"
        }
    }

    private var syncSymbol: String {
        if workspace?.isReviewBEnabled == true || workspace?.isReviewAExactlySynchronized == true {
            return "checkmark.circle.fill"
        }
        return "clock.badge.exclamationmark"
    }

    private var syncColor: Color {
        if workspace?.isReviewBEnabled == true || workspace?.isReviewAExactlySynchronized == true {
            return .green
        }
        return .orange
    }

    private func layerLabel(_ layer: ReviewLayerID) -> String {
        layer.rawValue.replacingOccurrences(of: "_", with: " ").uppercased()
    }

    private func viewerLayerLabel(_ layer: ReviewViewerLayerID) -> String {
        switch layer {
        case .surface: return "Surface"
        case .wireframe: return "Wire"
        case .tracker: return "Tracker"
        case .pixelROI: return "Pixel ROI"
        case .exactSourceFrame: return "Exact frame"
        }
    }
}

private enum ReviewWorkspaceViewError: LocalizedError {
    case bridgeBlocked(ReviewWorkspaceSlot)
    case commandQueueLimit(ReviewWorkspaceSlot)
    case commandJobMismatch(ReviewWorkspaceSlot)
    case commandSequence(ReviewWorkspaceSlot)
    case commandAcceptanceOrder(ReviewWorkspaceSlot)

    var errorDescription: String? {
        let slot: ReviewWorkspaceSlot
        let detail: String
        switch self {
        case .bridgeBlocked(let value):
            slot = value
            detail = "is blocked after a bridge failure; reload is required"
        case .commandQueueLimit(let value):
            slot = value
            detail = "exceeded the bounded 256-command snapshot"
        case .commandJobMismatch(let value):
            slot = value
            detail = "received a command for another job"
        case .commandSequence(let value):
            slot = value
            detail = "received a non-increasing command sequence"
        case .commandAcceptanceOrder(let value):
            slot = value
            detail = "accepted a command outside FIFO order"
        }
        return "Review viewer \(slot.rawValue.uppercased()) \(detail)."
    }
}

private struct ViewerLayerVisibility {
    private var surface = true
    private var wireframe = false
    private var tracker = true
    private var pixelROI = true
    private var exactSourceFrame = true

    func isVisible(_ layer: ReviewViewerLayerID) -> Bool {
        switch layer {
        case .surface: return surface
        case .wireframe: return wireframe
        case .tracker: return tracker
        case .pixelROI: return pixelROI
        case .exactSourceFrame: return exactSourceFrame
        }
    }

    mutating func set(_ layer: ReviewViewerLayerID, visible: Bool) {
        switch layer {
        case .surface: surface = visible
        case .wireframe: wireframe = visible
        case .tracker: tracker = visible
        case .pixelROI: pixelROI = visible
        case .exactSourceFrame: exactSourceFrame = visible
        }
    }
}
