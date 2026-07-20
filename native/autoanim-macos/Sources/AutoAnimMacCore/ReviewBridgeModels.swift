import Foundation

private struct ReviewAnyCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init?(stringValue: String) {
        self.stringValue = stringValue
        intValue = nil
    }

    init?(intValue: Int) {
        stringValue = String(intValue)
        self.intValue = intValue
    }
}

private func reviewRequireExactKeys(
    _ decoder: Decoder,
    _ expected: Set<String>
) throws {
    let container = try decoder.container(keyedBy: ReviewAnyCodingKey.self)
    let actual = Set(container.allKeys.map(\.stringValue))
    guard actual == expected else {
        throw ReviewModelError.invalid(
            field: decoder.codingPath.map(\.stringValue).joined(separator: "."),
            reason: "message fields differ from the fixed bridge contract"
        )
    }
}

public enum ReviewBridgeDirection: Sendable {
    case nativeToViewer
    case viewerToNative
}

public enum ReviewBridgeMessageType: String, Codable, CaseIterable, Sendable {
    case cursorSet = "cursor.set"
    case layerSet = "layer.set"
    case selectionSet = "selection.set"
    case revisionSet = "revision.set"
    case cursorChanged = "cursor.changed"
    case cursorApplied = "cursor.applied"
    case layerChanged = "layer.changed"
    case selectionChanged = "selection.changed"
    case revisionReady = "revision.ready"
    case viewerReady = "viewer.ready"
    case viewerError = "viewer.error"

    public var direction: ReviewBridgeDirection {
        switch self {
        case .cursorSet, .layerSet, .selectionSet, .revisionSet:
            return .nativeToViewer
        case .cursorChanged, .cursorApplied, .layerChanged, .selectionChanged,
             .revisionReady, .viewerReady, .viewerError:
            return .viewerToNative
        }
    }
}

public struct ReviewDecimalInteger: Codable, Equatable, Hashable, Sendable {
    public let rawValue: String

    public init(validating value: String, field: String = "decimal") throws {
        let body = value.hasPrefix("-") ? String(value.dropFirst()) : value
        let canonicalDigits = body == "0"
            || (!body.isEmpty && body.first != "0" && body.allSatisfy(\.isNumber))
        try reviewRequire(
            value.utf8.count <= 32
                && canonicalDigits
                && body.allSatisfy { $0.isASCII && $0.isNumber },
            field: field,
            reason: "must be one bounded canonical decimal integer"
        )
        rawValue = value
    }

    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer().decode(String.self)
        try self.init(
            validating: value,
            field: decoder.codingPath.map(\.stringValue).joined(separator: ".")
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    var int64Value: Int64? { Int64(rawValue) }
}

public struct ReviewBridgeTick: Codable, Equatable, Sendable {
    public let numerator: ReviewDecimalInteger
    public let denominator: ReviewDecimalInteger

    public init(_ numerator: ReviewDecimalInteger, _ denominator: ReviewDecimalInteger) throws {
        self.numerator = numerator
        self.denominator = denominator
        try validate()
    }

    public init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        numerator = try container.decode(ReviewDecimalInteger.self)
        denominator = try container.decode(ReviewDecimalInteger.self)
        guard container.isAtEnd else {
            throw ReviewModelError.invalid(
                field: decoder.codingPath.map(\.stringValue).joined(separator: "."),
                reason: "project tick must contain exactly two decimal strings"
            )
        }
        try validate()
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        try container.encode(numerator)
        try container.encode(denominator)
    }

    private func validate() throws {
        guard let numeratorValue = numerator.int64Value,
              let denominatorValue = denominator.int64Value
        else {
            throw ReviewModelError.invalid(
                field: "projectTick",
                reason: "project tick exceeds the native exact-integer range"
            )
        }
        try ReviewRational(numeratorValue, denominatorValue).validate(
            field: "projectTick"
        )
    }
}

public enum ReviewCursorOperation: String, Codable, CaseIterable, Sendable {
    case seek
    case step
    case pause
}

public enum ReviewBridgeVerification: String, Codable, CaseIterable, Sendable {
    case serverDecoded = "server_decoded"
    case presentedFrame = "presented_frame"
    case fallback
}

public enum ReviewCursorReason: String, Codable, CaseIterable, Sendable {
    case playback
    case seek
}

public enum ReviewViewerLayerID: String, Codable, CaseIterable, Sendable {
    case surface
    case wireframe
    case tracker
    case pixelROI
    case exactSourceFrame
}

public enum ReviewRegionSelection: String, Codable, CaseIterable, Sendable {
    case none
    case mouth
    case eyes
    case upperFace
    case head
}

public enum ReviewCameraPreset: String, Codable, CaseIterable, Sendable {
    case home
    case front
}

public enum ReviewSelection: Equatable, Sendable {
    case region(ReviewRegionSelection)
    case cameraPreset(ReviewCameraPreset)
}

extension ReviewSelection: Codable {
    private enum CodingKeys: String, CodingKey, CaseIterable {
        case kind
        case value
    }

    public init(from decoder: Decoder) throws {
        try reviewRequireExactKeys(decoder, Set(CodingKeys.allCases.map(\.rawValue)))
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(String.self, forKey: .kind)
        switch kind {
        case "region":
            self = .region(try container.decode(ReviewRegionSelection.self, forKey: .value))
        case "cameraPreset":
            self = .cameraPreset(try container.decode(ReviewCameraPreset.self, forKey: .value))
        default:
            throw ReviewModelError.invalid(field: "selection.kind", reason: "unsupported selection kind")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .region(let value):
            try container.encode("region", forKey: .kind)
            try container.encode(value, forKey: .value)
        case .cameraPreset(let value):
            try container.encode("cameraPreset", forKey: .kind)
            try container.encode(value, forKey: .value)
        }
    }
}

public enum ReviewViewerCapability: String, Codable, CaseIterable, Sendable {
    case layer
    case selection
    case revision
    case cursor
}

public enum ReviewViewerErrorCode: String, Codable, CaseIterable, Sendable {
    case bridgeEnvelopeInvalid = "BRIDGE_ENVELOPE_INVALID"
    case bridgePayloadInvalid = "BRIDGE_PAYLOAD_INVALID"
    case diagnosticsLoadFailed = "DIAGNOSTICS_LOAD_FAILED"
    case evidenceLoadFailed = "EVIDENCE_LOAD_FAILED"
    case exactFrameLoadFailed = "EXACT_FRAME_LOAD_FAILED"
    case glbLoadFailed = "GLB_LOAD_FAILED"
}

public struct ReviewCursorSetPayload: Codable, Equatable, Sendable {
    public let frameIndex: Int
    public let expectedSourcePTS: ReviewDecimalInteger
    public let operation: ReviewCursorOperation

    enum CodingKeys: String, CodingKey, CaseIterable {
        case frameIndex
        case expectedSourcePTS
        case operation
    }

    public init(frameIndex: Int, expectedSourcePTS: ReviewDecimalInteger, operation: ReviewCursorOperation) throws {
        try reviewRequire((0..<ReviewContract.maximumFrames).contains(frameIndex), field: "cursor.set.frameIndex", reason: "frame is outside U1 bounds")
        self.frameIndex = frameIndex
        self.expectedSourcePTS = expectedSourcePTS
        self.operation = operation
    }

    public init(from decoder: Decoder) throws {
        try reviewRequireExactKeys(decoder, Set(CodingKeys.allCases.map(\.rawValue)))
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            frameIndex: container.decode(Int.self, forKey: .frameIndex),
            expectedSourcePTS: container.decode(ReviewDecimalInteger.self, forKey: .expectedSourcePTS),
            operation: container.decode(ReviewCursorOperation.self, forKey: .operation)
        )
    }
}

public struct ReviewCursorChangedPayload: Codable, Equatable, Sendable {
    public let frameIndex: Int
    public let sourcePTS: ReviewDecimalInteger
    public let projectTick: ReviewBridgeTick
    public let verification: ReviewBridgeVerification
    public let reason: ReviewCursorReason

    enum CodingKeys: String, CodingKey, CaseIterable {
        case frameIndex
        case sourcePTS
        case projectTick
        case verification
        case reason
    }

    public init(frameIndex: Int, sourcePTS: ReviewDecimalInteger, projectTick: ReviewBridgeTick, verification: ReviewBridgeVerification, reason: ReviewCursorReason) throws {
        try reviewRequire((0..<ReviewContract.maximumFrames).contains(frameIndex), field: "cursor.changed.frameIndex", reason: "frame is outside U1 bounds")
        self.frameIndex = frameIndex
        self.sourcePTS = sourcePTS
        self.projectTick = projectTick
        self.verification = verification
        self.reason = reason
    }

    public init(from decoder: Decoder) throws {
        try reviewRequireExactKeys(decoder, Set(CodingKeys.allCases.map(\.rawValue)))
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            frameIndex: container.decode(Int.self, forKey: .frameIndex),
            sourcePTS: container.decode(ReviewDecimalInteger.self, forKey: .sourcePTS),
            projectTick: container.decode(ReviewBridgeTick.self, forKey: .projectTick),
            verification: container.decode(ReviewBridgeVerification.self, forKey: .verification),
            reason: container.decode(ReviewCursorReason.self, forKey: .reason)
        )
    }
}

public struct ReviewCursorAppliedPayload: Codable, Equatable, Sendable {
    public let requestSequence: UInt64
    public let frameIndex: Int
    public let sourcePTS: ReviewDecimalInteger
    public let verification: ReviewBridgeVerification

    enum CodingKeys: String, CodingKey, CaseIterable {
        case requestSequence
        case frameIndex
        case sourcePTS
        case verification
    }

    public init(requestSequence: UInt64, frameIndex: Int, sourcePTS: ReviewDecimalInteger, verification: ReviewBridgeVerification) throws {
        try reviewRequire(requestSequence <= ReviewBridgeEnvelope.maximumSequence, field: "cursor.applied.requestSequence", reason: "must be a uint53")
        try reviewRequire((0..<ReviewContract.maximumFrames).contains(frameIndex), field: "cursor.applied.frameIndex", reason: "frame is outside U1 bounds")
        try reviewRequire(verification == .serverDecoded, field: "cursor.applied.verification", reason: "exact cursor acknowledgement must be server_decoded")
        self.requestSequence = requestSequence
        self.frameIndex = frameIndex
        self.sourcePTS = sourcePTS
        self.verification = verification
    }

    public init(from decoder: Decoder) throws {
        try reviewRequireExactKeys(decoder, Set(CodingKeys.allCases.map(\.rawValue)))
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            requestSequence: container.decode(UInt64.self, forKey: .requestSequence),
            frameIndex: container.decode(Int.self, forKey: .frameIndex),
            sourcePTS: container.decode(ReviewDecimalInteger.self, forKey: .sourcePTS),
            verification: container.decode(ReviewBridgeVerification.self, forKey: .verification)
        )
    }
}

public struct ReviewLayerPayload: Codable, Equatable, Sendable {
    public let layerID: ReviewViewerLayerID
    public let visible: Bool

    enum CodingKeys: String, CodingKey, CaseIterable {
        case layerID
        case visible
    }

    public init(layerID: ReviewViewerLayerID, visible: Bool) {
        self.layerID = layerID
        self.visible = visible
    }

    public init(from decoder: Decoder) throws {
        try reviewRequireExactKeys(decoder, Set(CodingKeys.allCases.map(\.rawValue)))
        let container = try decoder.container(keyedBy: CodingKeys.self)
        layerID = try container.decode(ReviewViewerLayerID.self, forKey: .layerID)
        visible = try container.decode(Bool.self, forKey: .visible)
    }
}

public struct ReviewRevisionPayload: Codable, Equatable, Sendable {
    public let comparisonKey: String
    public let revisionID: String

    enum CodingKeys: String, CodingKey, CaseIterable {
        case comparisonKey
        case revisionID
    }

    public init(comparisonKey: String, revisionID: String) throws {
        try reviewRequire(ReviewValidation.isSHA256(comparisonKey), field: "comparisonKey", reason: "must be a lowercase SHA-256")
        try reviewRequire(ReviewValidation.isBridgeRevisionID(revisionID), field: "revisionID", reason: "must be a bounded canonical revision ID")
        self.comparisonKey = comparisonKey
        self.revisionID = revisionID
    }

    public init(from decoder: Decoder) throws {
        try reviewRequireExactKeys(decoder, Set(CodingKeys.allCases.map(\.rawValue)))
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            comparisonKey: container.decode(String.self, forKey: .comparisonKey),
            revisionID: container.decode(String.self, forKey: .revisionID)
        )
    }
}

public struct ReviewViewerReadyPayload: Codable, Equatable, Sendable {
    public let comparisonKey: String
    public let revisionID: String
    public let frameCount: Int
    public let capabilities: [ReviewViewerCapability]

    enum CodingKeys: String, CodingKey, CaseIterable {
        case comparisonKey
        case revisionID
        case frameCount
        case capabilities
    }

    public init(comparisonKey: String, revisionID: String, frameCount: Int, capabilities: [ReviewViewerCapability]) throws {
        try reviewRequire(ReviewValidation.isSHA256(comparisonKey), field: "viewer.ready.comparisonKey", reason: "must be a lowercase SHA-256")
        try reviewRequire(ReviewValidation.isBridgeRevisionID(revisionID), field: "viewer.ready.revisionID", reason: "must be a bounded canonical revision ID")
        try reviewRequire((1...ReviewContract.maximumFrames).contains(frameCount), field: "viewer.ready.frameCount", reason: "frame count is outside U1 bounds")
        let base: [ReviewViewerCapability] = [.layer, .selection, .revision]
        try reviewRequire(capabilities == base || capabilities == base + [.cursor], field: "viewer.ready.capabilities", reason: "capability list differs from the fixed viewer contract")
        self.comparisonKey = comparisonKey
        self.revisionID = revisionID
        self.frameCount = frameCount
        self.capabilities = capabilities
    }

    public init(from decoder: Decoder) throws {
        try reviewRequireExactKeys(decoder, Set(CodingKeys.allCases.map(\.rawValue)))
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            comparisonKey: container.decode(String.self, forKey: .comparisonKey),
            revisionID: container.decode(String.self, forKey: .revisionID),
            frameCount: container.decode(Int.self, forKey: .frameCount),
            capabilities: container.decode([ReviewViewerCapability].self, forKey: .capabilities)
        )
    }
}

public struct ReviewViewerErrorPayload: Codable, Equatable, Sendable {
    public let code: ReviewViewerErrorCode
    public let detail: String
    public let recoverable: Bool

    enum CodingKeys: String, CodingKey, CaseIterable {
        case code
        case detail
        case recoverable
    }

    public init(code: ReviewViewerErrorCode, detail: String, recoverable: Bool) throws {
        try reviewRequire(detail.utf8.count <= 512, field: "viewer.error.detail", reason: "detail exceeds 512 UTF-8 bytes")
        self.code = code
        self.detail = detail
        self.recoverable = recoverable
    }

    public init(from decoder: Decoder) throws {
        try reviewRequireExactKeys(decoder, Set(CodingKeys.allCases.map(\.rawValue)))
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            code: container.decode(ReviewViewerErrorCode.self, forKey: .code),
            detail: container.decode(String.self, forKey: .detail),
            recoverable: container.decode(Bool.self, forKey: .recoverable)
        )
    }
}

public enum ReviewBridgePayload: Equatable, Sendable {
    case cursorSet(ReviewCursorSetPayload)
    case layerSet(ReviewLayerPayload)
    case selectionSet(ReviewSelection)
    case revisionSet(ReviewRevisionPayload)
    case cursorChanged(ReviewCursorChangedPayload)
    case cursorApplied(ReviewCursorAppliedPayload)
    case layerChanged(ReviewLayerPayload)
    case selectionChanged(ReviewSelection)
    case revisionReady(ReviewRevisionPayload)
    case viewerReady(ReviewViewerReadyPayload)
    case viewerError(ReviewViewerErrorPayload)

    public var messageType: ReviewBridgeMessageType {
        switch self {
        case .cursorSet: return .cursorSet
        case .layerSet: return .layerSet
        case .selectionSet: return .selectionSet
        case .revisionSet: return .revisionSet
        case .cursorChanged: return .cursorChanged
        case .cursorApplied: return .cursorApplied
        case .layerChanged: return .layerChanged
        case .selectionChanged: return .selectionChanged
        case .revisionReady: return .revisionReady
        case .viewerReady: return .viewerReady
        case .viewerError: return .viewerError
        }
    }
}

public struct ReviewBridgeEnvelope: Codable, Equatable, Sendable {
    public static let schemaVersion = "autoanim.wk-review-bridge/1.0"
    public static let maximumSequence: UInt64 = 9_007_199_254_740_991
    public static let maximumBytes = 64 * 1_024

    public let schemaVersion: String
    public let sequence: UInt64
    public let type: ReviewBridgeMessageType
    public let jobID: String
    public let payload: ReviewBridgePayload

    enum CodingKeys: String, CodingKey, CaseIterable {
        case schemaVersion
        case sequence
        case type
        case jobID
        case payload
    }

    public init(sequence: UInt64, jobID: String, payload: ReviewBridgePayload) throws {
        self.schemaVersion = Self.schemaVersion
        self.sequence = sequence
        self.type = payload.messageType
        self.jobID = jobID
        self.payload = payload
        try validate()
    }

    public init(from decoder: Decoder) throws {
        try reviewRequireExactKeys(decoder, Set(CodingKeys.allCases.map(\.rawValue)))
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decode(String.self, forKey: .schemaVersion)
        sequence = try container.decode(UInt64.self, forKey: .sequence)
        type = try container.decode(ReviewBridgeMessageType.self, forKey: .type)
        jobID = try container.decode(String.self, forKey: .jobID)
        switch type {
        case .cursorSet:
            payload = .cursorSet(try container.decode(ReviewCursorSetPayload.self, forKey: .payload))
        case .layerSet:
            payload = .layerSet(try container.decode(ReviewLayerPayload.self, forKey: .payload))
        case .selectionSet:
            payload = .selectionSet(try container.decode(ReviewSelection.self, forKey: .payload))
        case .revisionSet:
            payload = .revisionSet(try container.decode(ReviewRevisionPayload.self, forKey: .payload))
        case .cursorChanged:
            payload = .cursorChanged(try container.decode(ReviewCursorChangedPayload.self, forKey: .payload))
        case .cursorApplied:
            payload = .cursorApplied(try container.decode(ReviewCursorAppliedPayload.self, forKey: .payload))
        case .layerChanged:
            payload = .layerChanged(try container.decode(ReviewLayerPayload.self, forKey: .payload))
        case .selectionChanged:
            payload = .selectionChanged(try container.decode(ReviewSelection.self, forKey: .payload))
        case .revisionReady:
            payload = .revisionReady(try container.decode(ReviewRevisionPayload.self, forKey: .payload))
        case .viewerReady:
            payload = .viewerReady(try container.decode(ReviewViewerReadyPayload.self, forKey: .payload))
        case .viewerError:
            payload = .viewerError(try container.decode(ReviewViewerErrorPayload.self, forKey: .payload))
        }
        try validate()
    }

    public func encode(to encoder: Encoder) throws {
        try validate()
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(sequence, forKey: .sequence)
        try container.encode(type, forKey: .type)
        try container.encode(jobID, forKey: .jobID)
        switch payload {
        case .cursorSet(let value): try container.encode(value, forKey: .payload)
        case .layerSet(let value): try container.encode(value, forKey: .payload)
        case .selectionSet(let value): try container.encode(value, forKey: .payload)
        case .revisionSet(let value): try container.encode(value, forKey: .payload)
        case .cursorChanged(let value): try container.encode(value, forKey: .payload)
        case .cursorApplied(let value): try container.encode(value, forKey: .payload)
        case .layerChanged(let value): try container.encode(value, forKey: .payload)
        case .selectionChanged(let value): try container.encode(value, forKey: .payload)
        case .revisionReady(let value): try container.encode(value, forKey: .payload)
        case .viewerReady(let value): try container.encode(value, forKey: .payload)
        case .viewerError(let value): try container.encode(value, forKey: .payload)
        }
    }

    public static func decode(
        _ data: Data,
        direction: ReviewBridgeDirection
    ) throws -> ReviewBridgeEnvelope {
        try reviewRequire(!data.isEmpty && data.count <= maximumBytes, field: "bridge", reason: "envelope is empty or exceeds 64 KiB")
        let envelope = try JSONDecoder().decode(ReviewBridgeEnvelope.self, from: data)
        try envelope.validate(direction: direction)
        return envelope
    }

    public func encoded() throws -> Data {
        let data = try JSONEncoder().encode(self)
        try reviewRequire(data.count <= Self.maximumBytes, field: "bridge", reason: "envelope exceeds 64 KiB")
        return data
    }

    public func validate(direction: ReviewBridgeDirection? = nil) throws {
        try reviewRequire(schemaVersion == Self.schemaVersion, field: "bridge.schemaVersion", reason: "unsupported schema")
        try reviewRequire(sequence <= Self.maximumSequence, field: "bridge.sequence", reason: "must be a uint53")
        try reviewRequire(ReviewValidation.isJobID(jobID), field: "bridge.jobID", reason: "must be a canonical AutoAnim job ID")
        try reviewRequire(type == payload.messageType, field: "bridge.type", reason: "does not match payload")
        if let direction {
            try reviewRequire(type.direction == direction, field: "bridge.type", reason: "message direction is not allowed")
        }
    }
}

public struct ReviewBridgeSequenceGate: Equatable, Sendable {
    public private(set) var lastAcceptedSequence: UInt64?

    public init(lastAcceptedSequence: UInt64? = nil) {
        self.lastAcceptedSequence = lastAcceptedSequence
    }

    public mutating func accept(_ envelope: ReviewBridgeEnvelope) -> Bool {
        if let lastAcceptedSequence, envelope.sequence <= lastAcceptedSequence {
            return false
        }
        lastAcceptedSequence = envelope.sequence
        return true
    }
}
