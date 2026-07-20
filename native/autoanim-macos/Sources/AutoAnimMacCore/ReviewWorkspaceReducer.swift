import Foundation

public enum ReviewWorkspaceSlot: String, Codable, CaseIterable, Sendable {
    case a
    case b
}

public struct ReviewFrameBinding: Equatable, Sendable {
    public let frameIndex: Int
    public let sourcePTS: Int64

    public init(frameIndex: Int, sourcePTS: Int64) {
        self.frameIndex = frameIndex
        self.sourcePTS = sourcePTS
    }
}

public struct ReviewWorkspaceCommand: Equatable, Sendable {
    public let slot: ReviewWorkspaceSlot
    public let envelope: ReviewBridgeEnvelope

    public init(slot: ReviewWorkspaceSlot, envelope: ReviewBridgeEnvelope) {
        self.slot = slot
        self.envelope = envelope
    }
}

public enum ReviewWorkspaceReceiveResult: Equatable, Sendable {
    case accepted
    case acceptedExactCursor(ReviewFrameBinding)
    case viewerError(ReviewViewerErrorPayload)
    case ignoredStale
    case ignoredUnmatchedAcknowledgement
}

public enum ReviewFrameSynchronizationState: Equatable, Sendable {
    case idle
    case pending(ReviewFrameBinding)
    case exact(ReviewFrameBinding)
}

public enum ReviewWorkspaceError: Error, Equatable, LocalizedError, Sendable {
    case comparisonKeyMismatch
    case withinJobComparisonUnsupported
    case comparisonUnavailable
    case viewerNotReady(ReviewWorkspaceSlot)
    case frameOutOfRange
    case wrongViewerJob
    case invalidViewerFrame
    case sequenceExhausted
    case comparisonNotSynchronized

    public var errorDescription: String? {
        switch self {
        case .comparisonKeyMismatch:
            return "Review B does not have the exact same comparison key as review A."
        case .withinJobComparisonUnsupported:
            return "ReviewBundle v1 does not support within-job A/B."
        case .comparisonUnavailable:
            return "No compatible cross-bundle review B is loaded."
        case .viewerNotReady(let slot):
            return "Review viewer \(slot.rawValue.uppercased()) is not ready for exact cursor commands."
        case .frameOutOfRange:
            return "The requested review frame is outside the exact source-PTS clock."
        case .wrongViewerJob:
            return "The bridge message belongs to another job."
        case .invalidViewerFrame:
            return "The bridge message is not bound to the bundle's exact frame and source PTS."
        case .sequenceExhausted:
            return "The uint53 bridge command sequence is exhausted."
        case .comparisonNotSynchronized:
            return "Review B remains disabled until both viewers acknowledge the exact frame and PTS."
        }
    }
}

public struct ReviewWorkspaceViewerState: Equatable, Sendable {
    public fileprivate(set) var sequenceGate = ReviewBridgeSequenceGate()
    public fileprivate(set) var nextOutboundSequence: UInt64 = 0
    public fileprivate(set) var outboundSequenceExhausted = false
    public fileprivate(set) var frameCount: Int?
    public fileprivate(set) var capabilities: [ReviewViewerCapability] = []
    public fileprivate(set) var revisionReady = false
    public fileprivate(set) var pendingCursor: ReviewWorkspacePendingCursor?
    public fileprivate(set) var exactCursor: ReviewFrameBinding?
    public fileprivate(set) var reportedCursor: ReviewFrameBinding?
    public fileprivate(set) var lastError: ReviewViewerErrorPayload?
    public fileprivate(set) var acknowledgedLayers: [ReviewLayerPayload] = []
    public fileprivate(set) var acknowledgedSelection: ReviewSelection?

    public init() {}

    public var supportsExactCursor: Bool {
        frameCount != nil && capabilities.contains(.cursor) && revisionReady
    }

    public var hasCompleteAcknowledgedLayerState: Bool {
        acknowledgedLayers.count == ReviewViewerLayerID.allCases.count
            && acknowledgedLayers.map(\.layerID) == ReviewViewerLayerID.allCases
    }
}

public struct ReviewWorkspacePendingCursor: Equatable, Sendable {
    public let requestSequence: UInt64
    public let binding: ReviewFrameBinding
}

public struct ReviewWorkspaceState: Equatable, Sendable {
    public let reviewA: ReviewBundle
    public private(set) var reviewB: ReviewBundle?
    public private(set) var activeSlot: ReviewWorkspaceSlot = .a
    public private(set) var targetFrame: ReviewFrameBinding?
    public private(set) var viewerA = ReviewWorkspaceViewerState()
    public private(set) var viewerB = ReviewWorkspaceViewerState()

    public init(reviewA: ReviewBundle) throws {
        try reviewA.validate()
        self.reviewA = reviewA
    }

    public var supportsWithinJobAB: Bool { false }
    public var supportsCorrectionAuthoring: Bool { false }

    public var comparisonKeysMatchExactly: Bool {
        guard let reviewB else { return false }
        return reviewA.comparisonKey == reviewB.comparisonKey
    }

    public var isReviewBEnabled: Bool {
        guard comparisonKeysMatchExactly,
              let targetFrame,
              viewerA.exactCursor == targetFrame,
              viewerB.exactCursor == targetFrame,
              crossBundleRenderStateMatches
        else {
            return false
        }
        return true
    }

    public var crossBundleRenderStateMatches: Bool {
        guard viewerA.hasCompleteAcknowledgedLayerState,
              viewerB.hasCompleteAcknowledgedLayerState,
              let selectionA = viewerA.acknowledgedSelection,
              let selectionB = viewerB.acknowledgedSelection
        else {
            return false
        }
        return viewerA.acknowledgedLayers == viewerB.acknowledgedLayers
            && selectionA == selectionB
    }

    public var cameraOrbitComparisonVerified: Bool { false }

    public var reviewAFrameSynchronization: ReviewFrameSynchronizationState {
        frameSynchronization(viewerA)
    }

    public var isReviewAExactlySynchronized: Bool {
        if case .exact = reviewAFrameSynchronization { return true }
        return false
    }

    public mutating func loadReviewB(_ bundle: ReviewBundle) throws {
        try bundle.validate()
        guard bundle.sourceManifest.jobID != reviewA.sourceManifest.jobID else {
            throw ReviewWorkspaceError.withinJobComparisonUnsupported
        }
        guard bundle.comparisonKey == reviewA.comparisonKey else {
            throw ReviewWorkspaceError.comparisonKeyMismatch
        }
        reviewB = bundle
        activeSlot = .a
        targetFrame = nil
        viewerB = ReviewWorkspaceViewerState()
        viewerA.pendingCursor = nil
        viewerA.exactCursor = nil
    }

    public mutating func unloadReviewB() {
        reviewB = nil
        activeSlot = .a
        targetFrame = nil
        viewerB = ReviewWorkspaceViewerState()
        viewerA.pendingCursor = nil
        viewerA.exactCursor = nil
    }

    public mutating func resetViewerDocument(for slot: ReviewWorkspaceSlot) {
        if slot == .a {
            viewerA = ReviewWorkspaceViewerState()
        } else {
            viewerB = ReviewWorkspaceViewerState()
        }
        targetFrame = nil
        viewerA.pendingCursor = nil
        viewerA.exactCursor = nil
        viewerB.pendingCursor = nil
        viewerB.exactCursor = nil
        activeSlot = .a
    }

    public mutating func requestComparisonFrame(
        _ frameIndex: Int,
        operation: ReviewCursorOperation = .seek
    ) throws -> [ReviewWorkspaceCommand] {
        guard let reviewB else { throw ReviewWorkspaceError.comparisonUnavailable }
        guard reviewB.comparisonKey == reviewA.comparisonKey else {
            throw ReviewWorkspaceError.comparisonKeyMismatch
        }
        guard reviewA.clock.sourcePTS.indices.contains(frameIndex),
              reviewB.clock.sourcePTS.indices.contains(frameIndex),
              reviewA.clock.sourcePTS[frameIndex] == reviewB.clock.sourcePTS[frameIndex]
        else {
            throw ReviewWorkspaceError.frameOutOfRange
        }
        guard viewerA.supportsExactCursor else {
            throw ReviewWorkspaceError.viewerNotReady(.a)
        }
        guard viewerB.supportsExactCursor else {
            throw ReviewWorkspaceError.viewerNotReady(.b)
        }
        guard !viewerA.outboundSequenceExhausted,
              !viewerB.outboundSequenceExhausted
        else {
            throw ReviewWorkspaceError.sequenceExhausted
        }
        return [
            try requestFrame(frameIndex, for: .a, operation: operation),
            try requestFrame(frameIndex, for: .b, operation: operation),
        ]
    }

    public mutating func requestFrame(
        _ frameIndex: Int,
        for slot: ReviewWorkspaceSlot,
        operation: ReviewCursorOperation = .seek
    ) throws -> ReviewWorkspaceCommand {
        let bundle = try bundle(for: slot)
        guard bundle.clock.sourcePTS.indices.contains(frameIndex) else {
            throw ReviewWorkspaceError.frameOutOfRange
        }
        let viewer = slot == .a ? viewerA : viewerB
        guard viewer.supportsExactCursor else {
            throw ReviewWorkspaceError.viewerNotReady(slot)
        }
        guard !viewer.outboundSequenceExhausted else {
            throw ReviewWorkspaceError.sequenceExhausted
        }
        let binding = ReviewFrameBinding(
            frameIndex: frameIndex,
            sourcePTS: bundle.clock.sourcePTS[frameIndex]
        )
        if targetFrame != binding {
            viewerA.pendingCursor = nil
            viewerA.exactCursor = nil
            viewerB.pendingCursor = nil
            viewerB.exactCursor = nil
        }
        targetFrame = binding
        let decimal = try ReviewDecimalInteger(validating: String(binding.sourcePTS))
        let payload = try ReviewCursorSetPayload(
            frameIndex: binding.frameIndex,
            expectedSourcePTS: decimal,
            operation: operation
        )
        let command = try makeCommand(
            slot: slot,
            bundle: bundle,
            payload: .cursorSet(payload)
        )
        let pending = ReviewWorkspacePendingCursor(
            requestSequence: command.envelope.sequence,
            binding: binding
        )
        if slot == .a {
            viewerA.pendingCursor = pending
            viewerA.exactCursor = nil
        } else {
            viewerB.pendingCursor = pending
            viewerB.exactCursor = nil
        }
        return command
    }

    public mutating func requestLayer(
        _ layerID: ReviewViewerLayerID,
        visible: Bool,
        for slot: ReviewWorkspaceSlot
    ) throws -> ReviewWorkspaceCommand {
        let bundle = try bundle(for: slot)
        let viewer = slot == .a ? viewerA : viewerB
        guard viewer.frameCount != nil,
              viewer.revisionReady,
              viewer.capabilities.contains(.layer)
        else {
            throw ReviewWorkspaceError.viewerNotReady(slot)
        }
        return try makeCommand(
            slot: slot,
            bundle: bundle,
            payload: .layerSet(ReviewLayerPayload(layerID: layerID, visible: visible))
        )
    }

    public mutating func requestSelection(
        _ selection: ReviewSelection,
        for slot: ReviewWorkspaceSlot
    ) throws -> ReviewWorkspaceCommand {
        let bundle = try bundle(for: slot)
        let viewer = slot == .a ? viewerA : viewerB
        guard viewer.frameCount != nil,
              viewer.revisionReady,
              viewer.capabilities.contains(.selection)
        else {
            throw ReviewWorkspaceError.viewerNotReady(slot)
        }
        return try makeCommand(
            slot: slot,
            bundle: bundle,
            payload: .selectionSet(selection)
        )
    }

    public mutating func select(_ slot: ReviewWorkspaceSlot) throws {
        if slot == .b && !isReviewBEnabled {
            throw ReviewWorkspaceError.comparisonNotSynchronized
        }
        activeSlot = slot
    }

    public mutating func receive(
        _ envelope: ReviewBridgeEnvelope,
        from slot: ReviewWorkspaceSlot
    ) throws -> ReviewWorkspaceReceiveResult {
        try envelope.validate(direction: .viewerToNative)
        let bundle = try bundle(for: slot)
        guard envelope.jobID == bundle.sourceManifest.jobID else {
            throw ReviewWorkspaceError.wrongViewerJob
        }
        var viewer = slot == .a ? viewerA : viewerB
        guard viewer.sequenceGate.accept(envelope) else { return .ignoredStale }
        let result = try apply(envelope.payload, bundle: bundle, viewer: &viewer)
        if slot == .a { viewerA = viewer } else { viewerB = viewer }
        return result
    }

    private func bundle(for slot: ReviewWorkspaceSlot) throws -> ReviewBundle {
        switch slot {
        case .a:
            return reviewA
        case .b:
            guard let reviewB else { throw ReviewWorkspaceError.comparisonUnavailable }
            return reviewB
        }
    }

    private func frameSynchronization(
        _ viewer: ReviewWorkspaceViewerState
    ) -> ReviewFrameSynchronizationState {
        if let targetFrame, viewer.exactCursor == targetFrame {
            return .exact(targetFrame)
        }
        if let pending = viewer.pendingCursor {
            return .pending(pending.binding)
        }
        return .idle
    }

    private mutating func makeCommand(
        slot: ReviewWorkspaceSlot,
        bundle: ReviewBundle,
        payload: ReviewBridgePayload
    ) throws -> ReviewWorkspaceCommand {
        var viewer = slot == .a ? viewerA : viewerB
        guard !viewer.outboundSequenceExhausted else {
            throw ReviewWorkspaceError.sequenceExhausted
        }
        let sequence = viewer.nextOutboundSequence
        if sequence == ReviewBridgeEnvelope.maximumSequence {
            viewer.outboundSequenceExhausted = true
        } else {
            viewer.nextOutboundSequence += 1
        }
        let envelope = try ReviewBridgeEnvelope(
            sequence: sequence,
            jobID: bundle.sourceManifest.jobID,
            payload: payload
        )
        if slot == .a { viewerA = viewer } else { viewerB = viewer }
        return ReviewWorkspaceCommand(slot: slot, envelope: envelope)
    }

    private func frameBinding(
        index: Int,
        sourcePTS: ReviewDecimalInteger,
        bundle: ReviewBundle
    ) throws -> ReviewFrameBinding {
        guard bundle.clock.sourcePTS.indices.contains(index),
              sourcePTS.rawValue == String(bundle.clock.sourcePTS[index])
        else {
            throw ReviewWorkspaceError.invalidViewerFrame
        }
        return ReviewFrameBinding(frameIndex: index, sourcePTS: bundle.clock.sourcePTS[index])
    }

    private func validateRevisionBinding(
        comparisonKey: String,
        revisionID: String,
        bundle: ReviewBundle
    ) throws {
        guard comparisonKey == bundle.comparisonKey.comparisonKeySHA256,
              bundle.revisionGraph.renderableRevisions.count == 1,
              revisionID == bundle.revisionGraph.renderableRevisions[0].revisionID
        else {
            throw ReviewWorkspaceError.invalidViewerFrame
        }
    }

    private mutating func apply(
        _ payload: ReviewBridgePayload,
        bundle: ReviewBundle,
        viewer: inout ReviewWorkspaceViewerState
    ) throws -> ReviewWorkspaceReceiveResult {
        switch payload {
        case .viewerReady(let ready):
            try validateRevisionBinding(
                comparisonKey: ready.comparisonKey,
                revisionID: ready.revisionID,
                bundle: bundle
            )
            guard ready.frameCount == bundle.clock.frameCount else {
                throw ReviewWorkspaceError.invalidViewerFrame
            }
            viewer.frameCount = ready.frameCount
            viewer.capabilities = ready.capabilities
        case .revisionReady(let revision):
            try validateRevisionBinding(
                comparisonKey: revision.comparisonKey,
                revisionID: revision.revisionID,
                bundle: bundle
            )
            viewer.revisionReady = true
        case .cursorChanged(let cursor):
            let binding = try frameBinding(
                index: cursor.frameIndex,
                sourcePTS: cursor.sourcePTS,
                bundle: bundle
            )
            viewer.reportedCursor = binding
            if viewer.exactCursor != binding || cursor.verification != .serverDecoded {
                viewer.exactCursor = nil
            }
        case .cursorApplied(let acknowledgement):
            let binding = try frameBinding(
                index: acknowledgement.frameIndex,
                sourcePTS: acknowledgement.sourcePTS,
                bundle: bundle
            )
            guard let pending = viewer.pendingCursor,
                  pending.requestSequence == acknowledgement.requestSequence,
                  pending.binding == binding,
                  targetFrame == binding
            else {
                return .ignoredUnmatchedAcknowledgement
            }
            viewer.pendingCursor = nil
            viewer.reportedCursor = binding
            viewer.exactCursor = binding
            return .acceptedExactCursor(binding)
        case .viewerError(let error):
            viewer.lastError = error
            viewer.pendingCursor = nil
            viewer.exactCursor = nil
            return .viewerError(error)
        case .layerChanged(let layer):
            viewer.acknowledgedLayers.removeAll { $0.layerID == layer.layerID }
            viewer.acknowledgedLayers.append(layer)
            viewer.acknowledgedLayers.sort {
                let left = ReviewViewerLayerID.allCases.firstIndex(of: $0.layerID) ?? Int.max
                let right = ReviewViewerLayerID.allCases.firstIndex(of: $1.layerID) ?? Int.max
                return left < right
            }
        case .selectionChanged(let selection):
            viewer.acknowledgedSelection = selection
        case .cursorSet, .layerSet, .selectionSet, .revisionSet:
            throw ReviewModelError.invalid(
                field: "bridge.type",
                reason: "native command received as viewer event"
            )
        }
        return .accepted
    }
}
