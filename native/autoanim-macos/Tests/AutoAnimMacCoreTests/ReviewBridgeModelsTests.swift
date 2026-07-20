import Foundation
import Testing
@testable import AutoAnimMacCore

@Suite("Bounded WK review bridge")
struct ReviewBridgeModelsTests {
    private func decimal(_ value: Int64) throws -> ReviewDecimalInteger {
        try ReviewDecimalInteger(validating: String(value))
    }

    private func revisionBinding(
        jobID: String = ReviewFixture.jobA
    ) throws -> (comparisonKey: String, revisionID: String) {
        let bundle = try ReviewFixture.bundle(jobID: jobID)
        return (
            bundle.comparisonKey.comparisonKeySHA256,
            bundle.revisionGraph.renderableRevisions[0].revisionID
        )
    }

    @Test("Round-trips the fixed cursor command envelope")
    func cursorCommandRoundTrip() throws {
        let payload = try ReviewCursorSetPayload(
            frameIndex: 1,
            expectedSourcePTS: decimal(1_001),
            operation: .seek
        )
        let envelope = try ReviewBridgeEnvelope(
            sequence: ReviewBridgeEnvelope.maximumSequence,
            jobID: ReviewFixture.jobA,
            payload: .cursorSet(payload)
        )
        let decoded = try ReviewBridgeEnvelope.decode(
            envelope.encoded(),
            direction: .nativeToViewer
        )
        #expect(decoded == envelope)
        #expect(decoded.schemaVersion == "autoanim.wk-review-bridge/1.0")
        #expect(decoded.type == .cursorSet)
        #expect(throws: ReviewModelError.self) {
            try ReviewBridgeEnvelope.decode(envelope.encoded(), direction: .viewerToNative)
        }
    }

    @Test("Decodes exact cursor acknowledgement with a uint53 request sequence")
    func exactCursorAcknowledgement() throws {
        let payload = try ReviewCursorAppliedPayload(
            requestSequence: ReviewBridgeEnvelope.maximumSequence,
            frameIndex: 2,
            sourcePTS: decimal(2_002),
            verification: .serverDecoded
        )
        let envelope = try ReviewBridgeEnvelope(
            sequence: 7,
            jobID: ReviewFixture.jobA,
            payload: .cursorApplied(payload)
        )
        let decoded = try ReviewBridgeEnvelope.decode(
            envelope.encoded(),
            direction: .viewerToNative
        )
        #expect(decoded.payload == .cursorApplied(payload))
    }

    @Test("Rejects non-uint53 sequences, unknown messages, and payload field injection")
    func rejectsUnboundedOrUnknownMessages() throws {
        let binding = try revisionBinding()
        let tooLarge = Data(
            "{\"schemaVersion\":\"autoanim.wk-review-bridge/1.0\",\"sequence\":9007199254740992,\"type\":\"revision.ready\",\"jobID\":\"\(ReviewFixture.jobA)\",\"payload\":{\"comparisonKey\":\"\(binding.comparisonKey)\",\"revisionID\":\"\(binding.revisionID)\"}}".utf8
        )
        #expect(throws: ReviewModelError.self) {
            try ReviewBridgeEnvelope.decode(tooLarge, direction: .viewerToNative)
        }

        let unknown = Data(
            #"{"schemaVersion":"autoanim.wk-review-bridge/1.0","sequence":0,"type":"script.execute","jobID":"01kxwwdq8gqrsrzycjc3c3kjy9","payload":{}}"#.utf8
        )
        #expect(throws: DecodingError.self) {
            try ReviewBridgeEnvelope.decode(unknown, direction: .viewerToNative)
        }

        let injected = Data(
            #"{"schemaVersion":"autoanim.wk-review-bridge/1.0","sequence":0,"type":"layer.changed","jobID":"01kxwwdq8gqrsrzycjc3c3kjy9","payload":{"layerID":"surface","visible":true,"script":"alert(1)"}}"#.utf8
        )
        #expect(throws: ReviewModelError.self) {
            try ReviewBridgeEnvelope.decode(injected, direction: .viewerToNative)
        }
    }

    @Test("Rejects invalid fixed payload enums, decimals, and sizes")
    func rejectsInvalidPayloads() throws {
        #expect(throws: ReviewModelError.self) {
            try ReviewDecimalInteger(validating: "01")
        }
        let invalidLayer = Data(
            #"{"schemaVersion":"autoanim.wk-review-bridge/1.0","sequence":0,"type":"layer.changed","jobID":"01kxwwdq8gqrsrzycjc3c3kjy9","payload":{"layerID":"arbitrary","visible":true}}"#.utf8
        )
        #expect(throws: DecodingError.self) {
            try ReviewBridgeEnvelope.decode(invalidLayer, direction: .viewerToNative)
        }
        let invalidVerification = Data(
            #"{"schemaVersion":"autoanim.wk-review-bridge/1.0","sequence":0,"type":"cursor.applied","jobID":"01kxwwdq8gqrsrzycjc3c3kjy9","payload":{"requestSequence":0,"frameIndex":0,"sourcePTS":"0","verification":"fallback"}}"#.utf8
        )
        #expect(throws: ReviewModelError.self) {
            try ReviewBridgeEnvelope.decode(invalidVerification, direction: .viewerToNative)
        }
        #expect(throws: ReviewModelError.self) {
            try ReviewBridgeEnvelope.decode(
                Data(repeating: 0x20, count: ReviewBridgeEnvelope.maximumBytes + 1),
                direction: .viewerToNative
            )
        }
        #expect(throws: ReviewModelError.self) {
            try ReviewViewerErrorPayload(
                code: .evidenceLoadFailed,
                detail: String(repeating: "x", count: 513),
                recoverable: true
            )
        }
    }

    @Test("Accepts only reduced project tick rationals")
    func validatesProjectTicks() throws {
        let valid = try ReviewBridgeTick(
            ReviewDecimalInteger(validating: "1001"),
            ReviewDecimalInteger(validating: "30000")
        )
        #expect(valid.numerator.rawValue == "1001")
        #expect(throws: ReviewModelError.self) {
            try ReviewBridgeTick(
                ReviewDecimalInteger(validating: "2"),
                ReviewDecimalInteger(validating: "60000")
            )
        }
        #expect(throws: ReviewModelError.self) {
            try ReviewBridgeTick(
                ReviewDecimalInteger(validating: "1"),
                ReviewDecimalInteger(validating: "0")
            )
        }
    }

    @Test("Sequence gate ignores stale and replayed viewer messages")
    func ignoresStaleSequences() throws {
        func ready(_ sequence: UInt64) throws -> ReviewBridgeEnvelope {
            let binding = try revisionBinding()
            return try ReviewBridgeEnvelope(
                sequence: sequence,
                jobID: ReviewFixture.jobA,
                payload: .viewerReady(
                    try ReviewViewerReadyPayload(
                        comparisonKey: binding.comparisonKey,
                        revisionID: binding.revisionID,
                        frameCount: 3,
                        capabilities: [.layer, .selection, .revision, .cursor]
                    )
                )
            )
        }
        var gate = ReviewBridgeSequenceGate()
        #expect(gate.accept(try ready(5)))
        #expect(!gate.accept(try ready(5)))
        #expect(!gate.accept(try ready(4)))
        #expect(gate.accept(try ready(6)))
        #expect(gate.lastAcceptedSequence == 6)
    }
}
