import Foundation
import Testing
@testable import AutoAnimMacCore

@Suite("Review workspace reducer")
struct ReviewWorkspaceReducerTests {
    private var defaultLayerState: [ReviewLayerPayload] {
        [
            ReviewLayerPayload(layerID: .surface, visible: true),
            ReviewLayerPayload(layerID: .wireframe, visible: false),
            ReviewLayerPayload(layerID: .tracker, visible: true),
            ReviewLayerPayload(layerID: .pixelROI, visible: true),
            ReviewLayerPayload(layerID: .exactSourceFrame, visible: true),
        ]
    }

    private func ready(
        jobID: String,
        frameCount: Int,
        sequence: UInt64 = 0
    ) throws -> ReviewBridgeEnvelope {
        let bundle = try ReviewFixture.bundle(jobID: jobID)
        return try ReviewBridgeEnvelope(
            sequence: sequence,
            jobID: jobID,
            payload: .viewerReady(
                try ReviewViewerReadyPayload(
                    comparisonKey: bundle.comparisonKey.comparisonKeySHA256,
                    revisionID: bundle.revisionGraph.renderableRevisions[0].revisionID,
                    frameCount: frameCount,
                    capabilities: [.layer, .selection, .revision, .cursor]
                )
            )
        )
    }

    private func revisionReady(
        jobID: String,
        sequence: UInt64 = 1
    ) throws -> ReviewBridgeEnvelope {
        let bundle = try ReviewFixture.bundle(jobID: jobID)
        return try ReviewBridgeEnvelope(
            sequence: sequence,
            jobID: jobID,
            payload: .revisionReady(
                try ReviewRevisionPayload(
                    comparisonKey: bundle.comparisonKey.comparisonKeySHA256,
                    revisionID: bundle.revisionGraph.renderableRevisions[0].revisionID
                )
            )
        )
    }

    private func acknowledgement(
        command: ReviewWorkspaceCommand,
        eventSequence: UInt64,
        requestSequence: UInt64? = nil,
        frameIndex: Int? = nil,
        sourcePTS: Int64? = nil
    ) throws -> ReviewBridgeEnvelope {
        guard case .cursorSet(let cursor) = command.envelope.payload else {
            Issue.record("Expected cursor.set command")
            throw ReviewWorkspaceError.invalidViewerFrame
        }
        let pts = sourcePTS.map(String.init) ?? cursor.expectedSourcePTS.rawValue
        return try ReviewBridgeEnvelope(
            sequence: eventSequence,
            jobID: command.envelope.jobID,
            payload: .cursorApplied(
                try ReviewCursorAppliedPayload(
                    requestSequence: requestSequence ?? command.envelope.sequence,
                    frameIndex: frameIndex ?? cursor.frameIndex,
                    sourcePTS: ReviewDecimalInteger(validating: pts),
                    verification: .serverDecoded
                )
            )
        )
    }

    private func prepare() throws -> ReviewWorkspaceState {
        let a = try ReviewFixture.bundle(jobID: ReviewFixture.jobA)
        let b = try ReviewFixture.bundle(jobID: ReviewFixture.jobB)
        var state = try ReviewWorkspaceState(reviewA: a)
        try state.loadReviewB(b)
        _ = try state.receive(ready(jobID: ReviewFixture.jobA, frameCount: 3), from: .a)
        _ = try state.receive(revisionReady(jobID: ReviewFixture.jobA), from: .a)
        _ = try state.receive(ready(jobID: ReviewFixture.jobB, frameCount: 3), from: .b)
        _ = try state.receive(revisionReady(jobID: ReviewFixture.jobB), from: .b)
        return state
    }

    @discardableResult
    private func acknowledgeRenderState(
        _ state: inout ReviewWorkspaceState,
        slot: ReviewWorkspaceSlot,
        layers: [ReviewLayerPayload]? = nil,
        selection: ReviewSelection = .region(.none),
        startingSequence: UInt64 = 3
    ) throws -> UInt64 {
        let jobID = slot == .a ? ReviewFixture.jobA : ReviewFixture.jobB
        var sequence = startingSequence
        for layer in layers ?? defaultLayerState {
            let envelope = try ReviewBridgeEnvelope(
                sequence: sequence,
                jobID: jobID,
                payload: .layerChanged(layer)
            )
            _ = try state.receive(envelope, from: slot)
            sequence += 1
        }
        let selectionEnvelope = try ReviewBridgeEnvelope(
            sequence: sequence,
            jobID: jobID,
            payload: .selectionChanged(selection)
        )
        _ = try state.receive(selectionEnvelope, from: slot)
        return sequence + 1
    }

    @Test("Requires a distinct bundle with an exactly equal comparison key")
    func requiresCompatibleCrossBundlePair() throws {
        let a = try ReviewFixture.bundle(jobID: ReviewFixture.jobA)
        var state = try ReviewWorkspaceState(reviewA: a)
        #expect(!state.supportsWithinJobAB)
        #expect(!state.supportsCorrectionAuthoring)

        #expect(throws: ReviewWorkspaceError.withinJobComparisonUnsupported) {
            try state.loadReviewB(ReviewFixture.bundle(jobID: ReviewFixture.jobA))
        }
        #expect(throws: ReviewWorkspaceError.comparisonKeyMismatch) {
            try state.loadReviewB(
                ReviewFixture.bundle(
                    jobID: ReviewFixture.jobB,
                    comparisonVariant: "different"
                )
            )
        }
        try state.loadReviewB(ReviewFixture.bundle(jobID: ReviewFixture.jobB))
        #expect(state.comparisonKeysMatchExactly)
        #expect(!state.isReviewBEnabled)
        #expect(throws: ReviewWorkspaceError.comparisonNotSynchronized) {
            try state.select(.b)
        }
    }

    @Test("Enables B only after both exact frame and PTS acknowledgements")
    func gatesComparisonOnExactAcknowledgements() throws {
        var state = try prepare()
        let commands = try state.requestComparisonFrame(1)
        #expect(commands.map(\.slot) == [.a, .b])
        #expect(!state.isReviewBEnabled)

        let aResult = try state.receive(
            acknowledgement(command: commands[0], eventSequence: 2),
            from: .a
        )
        #expect(aResult == .acceptedExactCursor(ReviewFrameBinding(frameIndex: 1, sourcePTS: 1_001)))
        #expect(!state.isReviewBEnabled)

        let bAck = try acknowledgement(command: commands[1], eventSequence: 2)
        let bResult = try state.receive(bAck, from: .b)
        #expect(bResult == .acceptedExactCursor(ReviewFrameBinding(frameIndex: 1, sourcePTS: 1_001)))
        #expect(!state.isReviewBEnabled)
        try acknowledgeRenderState(&state, slot: .a)
        try acknowledgeRenderState(&state, slot: .b)
        #expect(state.crossBundleRenderStateMatches)
        #expect(!state.cameraOrbitComparisonVerified)
        #expect(state.isReviewBEnabled)
        try state.select(.b)
        #expect(state.activeSlot == .b)

        #expect(try state.receive(bAck, from: .b) == .ignoredStale)
        #expect(state.isReviewBEnabled)
    }

    @Test("Ignores unmatched acknowledgements without enabling B")
    func ignoresUnmatchedAcknowledgements() throws {
        var state = try prepare()
        let commands = try state.requestComparisonFrame(2)
        let wrongRequest = try acknowledgement(
            command: commands[0],
            eventSequence: 2,
            requestSequence: commands[0].envelope.sequence + 1
        )
        #expect(try state.receive(wrongRequest, from: .a) == .ignoredUnmatchedAcknowledgement)
        #expect(!state.isReviewBEnabled)

        let correctA = try acknowledgement(command: commands[0], eventSequence: 3)
        _ = try state.receive(correctA, from: .a)
        _ = try state.receive(
            acknowledgement(command: commands[1], eventSequence: 2),
            from: .b
        )
        try acknowledgeRenderState(&state, slot: .a, startingSequence: 4)
        try acknowledgeRenderState(&state, slot: .b)
        #expect(state.isReviewBEnabled)
    }

    @Test("Rejects wrong jobs and frame bindings and ignores stale messages")
    func rejectsCrossJobOrWrongFrameMessages() throws {
        var state = try prepare()
        let commands = try state.requestComparisonFrame(1)
        let wrongJob = try ReviewBridgeEnvelope(
            sequence: 2,
            jobID: ReviewFixture.jobB,
            payload: .cursorApplied(
                try ReviewCursorAppliedPayload(
                    requestSequence: commands[0].envelope.sequence,
                    frameIndex: 1,
                    sourcePTS: ReviewDecimalInteger(validating: "1001"),
                    verification: .serverDecoded
                )
            )
        )
        #expect(throws: ReviewWorkspaceError.wrongViewerJob) {
            try state.receive(wrongJob, from: .a)
        }

        let wrongPTS = try acknowledgement(
            command: commands[0],
            eventSequence: 2,
            sourcePTS: 99
        )
        #expect(throws: ReviewWorkspaceError.invalidViewerFrame) {
            try state.receive(wrongPTS, from: .a)
        }

        let changed = try ReviewBridgeEnvelope(
            sequence: 1,
            jobID: ReviewFixture.jobA,
            payload: .layerChanged(ReviewLayerPayload(layerID: .surface, visible: false))
        )
        #expect(try state.receive(changed, from: .a) == .ignoredStale)
    }

    @Test("Moving either viewer away from the acknowledged frame disables B")
    func invalidatesSynchronizationWhenViewerMoves() throws {
        var state = try prepare()
        let commands = try state.requestComparisonFrame(1)
        _ = try state.receive(acknowledgement(command: commands[0], eventSequence: 2), from: .a)
        _ = try state.receive(acknowledgement(command: commands[1], eventSequence: 2), from: .b)
        try acknowledgeRenderState(&state, slot: .a)
        try acknowledgeRenderState(&state, slot: .b)
        #expect(state.isReviewBEnabled)

        let tick = try ReviewBridgeTick(
            ReviewDecimalInteger(validating: "0"),
            ReviewDecimalInteger(validating: "1")
        )
        let moved = try ReviewBridgeEnvelope(
            sequence: 9,
            jobID: ReviewFixture.jobB,
            payload: .cursorChanged(
                try ReviewCursorChangedPayload(
                    frameIndex: 0,
                    sourcePTS: ReviewDecimalInteger(validating: "0"),
                    projectTick: tick,
                    verification: .fallback,
                    reason: .playback
                )
            )
        )
        _ = try state.receive(moved, from: .b)
        #expect(!state.isReviewBEnabled)
        #expect(throws: ReviewWorkspaceError.comparisonNotSynchronized) {
            try state.select(.b)
        }
    }

    @Test("Will not issue exact cursor commands before both viewers are ready")
    func requiresExactCursorCapability() throws {
        let a = try ReviewFixture.bundle(jobID: ReviewFixture.jobA)
        let b = try ReviewFixture.bundle(jobID: ReviewFixture.jobB)
        var state = try ReviewWorkspaceState(reviewA: a)
        try state.loadReviewB(b)
        #expect(throws: ReviewWorkspaceError.viewerNotReady(.a)) {
            try state.requestComparisonFrame(0)
        }
    }

    @Test("A timeline requests and acknowledges an exact frame without loading B")
    func synchronizesSingleReviewA() throws {
        let a = try ReviewFixture.bundle(jobID: ReviewFixture.jobA)
        var state = try ReviewWorkspaceState(reviewA: a)
        _ = try state.receive(ready(jobID: ReviewFixture.jobA, frameCount: 3), from: .a)
        _ = try state.receive(revisionReady(jobID: ReviewFixture.jobA), from: .a)

        let command = try state.requestFrame(1, for: .a, operation: .step)
        #expect(command.slot == .a)
        #expect(command.envelope.jobID == ReviewFixture.jobA)
        #expect(state.reviewAFrameSynchronization == .pending(ReviewFrameBinding(frameIndex: 1, sourcePTS: 1_001)))
        #expect(!state.isReviewAExactlySynchronized)

        _ = try state.receive(
            acknowledgement(command: command, eventSequence: 2),
            from: .a
        )
        #expect(state.reviewAFrameSynchronization == .exact(ReviewFrameBinding(frameIndex: 1, sourcePTS: 1_001)))
        #expect(state.isReviewAExactlySynchronized)
        #expect(!state.isReviewBEnabled)
    }

    @Test("Reducer owns monotonic layer, selection, and cursor command sequences per slot")
    func ownsOutboundReviewCommandSequences() throws {
        var state = try prepare()
        let layer = try state.requestLayer(.pixelROI, visible: false, for: .a)
        let selection = try state.requestSelection(.region(.mouth), for: .a)
        let cursor = try state.requestFrame(2, for: .a)
        let bLayer = try state.requestLayer(.wireframe, visible: true, for: .b)

        #expect(layer.envelope.sequence == 0)
        #expect(selection.envelope.sequence == 1)
        #expect(cursor.envelope.sequence == 2)
        #expect(bLayer.envelope.sequence == 0)
        #expect(layer.envelope.jobID == ReviewFixture.jobA)
        #expect(selection.envelope.jobID == ReviewFixture.jobA)
        #expect(cursor.envelope.jobID == ReviewFixture.jobA)
        #expect(bLayer.envelope.jobID == ReviewFixture.jobB)
        #expect(layer.envelope.payload == .layerSet(ReviewLayerPayload(layerID: .pixelROI, visible: false)))
        #expect(selection.envelope.payload == .selectionSet(.region(.mouth)))
        #expect(bLayer.envelope.payload == .layerSet(ReviewLayerPayload(layerID: .wireframe, visible: true)))
    }

    @Test("Viewer readiness must bind the bundle comparison key and sole renderable revision")
    func rejectsWrongViewerRevisionBinding() throws {
        let a = try ReviewFixture.bundle(jobID: ReviewFixture.jobA)
        var state = try ReviewWorkspaceState(reviewA: a)
        let wrong = try ReviewBridgeEnvelope(
            sequence: 0,
            jobID: ReviewFixture.jobA,
            payload: .viewerReady(
                try ReviewViewerReadyPayload(
                    comparisonKey: ReviewFixture.digest("wrong-comparison"),
                    revisionID: a.revisionGraph.renderableRevisions[0].revisionID,
                    frameCount: a.clock.frameCount,
                    capabilities: [.layer, .selection, .revision, .cursor]
                )
            )
        )
        #expect(throws: ReviewWorkspaceError.invalidViewerFrame) {
            try state.receive(wrong, from: .a)
        }
    }

    @Test("Resetting a viewer document clears state and restarts both sequence domains")
    func resetsViewerDocumentLifecycle() throws {
        var state = try prepare()
        let firstCommands = try state.requestComparisonFrame(1)
        #expect(firstCommands[0].envelope.sequence == 0)
        #expect(firstCommands[1].envelope.sequence == 0)
        _ = try state.receive(acknowledgement(command: firstCommands[0], eventSequence: 2), from: .a)
        _ = try state.receive(acknowledgement(command: firstCommands[1], eventSequence: 2), from: .b)
        #expect(state.viewerA.exactCursor == ReviewFrameBinding(frameIndex: 1, sourcePTS: 1_001))
        #expect(state.viewerB.exactCursor == ReviewFrameBinding(frameIndex: 1, sourcePTS: 1_001))

        state.resetViewerDocument(for: .a)
        #expect(state.targetFrame == nil)
        #expect(state.viewerA.sequenceGate.lastAcceptedSequence == nil)
        #expect(state.viewerA.nextOutboundSequence == 0)
        #expect(state.viewerA.frameCount == nil)
        #expect(state.viewerA.capabilities.isEmpty)
        #expect(!state.viewerA.revisionReady)
        #expect(state.viewerA.pendingCursor == nil)
        #expect(state.viewerA.exactCursor == nil)
        #expect(state.viewerA.reportedCursor == nil)
        #expect(state.viewerA.lastError == nil)
        #expect(state.viewerA.acknowledgedLayers.isEmpty)
        #expect(state.viewerA.acknowledgedSelection == nil)
        #expect(state.viewerB.pendingCursor == nil)
        #expect(state.viewerB.exactCursor == nil)
        #expect(state.viewerB.frameCount == 3)
        #expect(state.viewerB.sequenceGate.lastAcceptedSequence == 2)
        #expect(state.reviewAFrameSynchronization == .idle)

        #expect(try state.receive(ready(jobID: ReviewFixture.jobA, frameCount: 3, sequence: 0), from: .a) == .accepted)
        #expect(try state.receive(revisionReady(jobID: ReviewFixture.jobA, sequence: 1), from: .a) == .accepted)
        let restarted = try state.requestFrame(1, for: .a)
        #expect(restarted.envelope.sequence == 0)
        #expect(restarted.envelope.jobID == ReviewFixture.jobA)
        #expect(state.reviewAFrameSynchronization == .pending(ReviewFrameBinding(frameIndex: 1, sourcePTS: 1_001)))
        _ = try state.receive(acknowledgement(command: restarted, eventSequence: 2), from: .a)
        #expect(state.reviewAFrameSynchronization == .exact(ReviewFrameBinding(frameIndex: 1, sourcePTS: 1_001)))
    }

    @Test("Viewer errors are typed, persisted, and cleared only by document reset")
    func surfacesPersistentViewerErrors() throws {
        var state = try prepare()
        let payload = try ReviewViewerErrorPayload(
            code: .exactFrameLoadFailed,
            detail: "Exact proxy frame could not be decoded.",
            recoverable: true
        )
        let error = try ReviewBridgeEnvelope(
            sequence: 2,
            jobID: ReviewFixture.jobA,
            payload: .viewerError(payload)
        )
        #expect(try state.receive(error, from: .a) == .viewerError(payload))
        #expect(state.viewerA.lastError == payload)

        let laterMessage = try ReviewBridgeEnvelope(
            sequence: 3,
            jobID: ReviewFixture.jobA,
            payload: .layerChanged(ReviewLayerPayload(layerID: .surface, visible: true))
        )
        #expect(try state.receive(laterMessage, from: .a) == .accepted)
        #expect(state.viewerA.lastError == payload)
        state.resetViewerDocument(for: .a)
        #expect(state.viewerA.lastError == nil)
    }

    @Test("B stays disabled until complete acknowledged layer and selection states match")
    func gatesBOnAcknowledgedRenderState() throws {
        var state = try prepare()
        let commands = try state.requestComparisonFrame(1)
        _ = try state.receive(acknowledgement(command: commands[0], eventSequence: 2), from: .a)
        _ = try state.receive(acknowledgement(command: commands[1], eventSequence: 2), from: .b)

        try acknowledgeRenderState(&state, slot: .a)
        var mismatched = defaultLayerState
        let wireframeIndex = mismatched.firstIndex { $0.layerID == .wireframe }!
        mismatched[wireframeIndex] = ReviewLayerPayload(layerID: .wireframe, visible: true)
        try acknowledgeRenderState(&state, slot: .b, layers: mismatched)
        #expect(!state.crossBundleRenderStateMatches)
        #expect(!state.isReviewBEnabled)

        let matchingWireframe = try ReviewBridgeEnvelope(
            sequence: 9,
            jobID: ReviewFixture.jobB,
            payload: .layerChanged(ReviewLayerPayload(layerID: .wireframe, visible: false))
        )
        _ = try state.receive(matchingWireframe, from: .b)
        #expect(state.crossBundleRenderStateMatches)
        #expect(state.isReviewBEnabled)
        #expect(!state.cameraOrbitComparisonVerified)
    }
}
