import CryptoKit
import Foundation

public enum ReviewModelError: Error, Equatable, LocalizedError, Sendable {
    case invalid(field: String, reason: String)

    public var errorDescription: String? {
        switch self {
        case .invalid(let field, let reason):
            return "Invalid ReviewBundle field \(field): \(reason)"
        }
    }
}

func reviewRequire(
    _ condition: @autoclosure () -> Bool,
    field: String,
    reason: String
) throws {
    guard condition() else {
        throw ReviewModelError.invalid(field: field, reason: reason)
    }
}

enum ReviewContract {
    static let bundleSchema = "autoanim.review-bundle/1.0"
    static let clockSchema = "autoanim.review-clock/1.0"
    static let comparisonSchema = "autoanim.review-comparison-key/1.0"
    static let layerSchema = "autoanim.review-layer/1.0"
    static let revisionGraphSchema = "autoanim.review-revision-graph/1.0"
    static let closeupSchema = "autoanim.review-closeup/1.0"
    static let materialSchema = "autoanim.review-material-channel/1.0"
    static let correctionSchema = "autoanim.review-correction-eligibility/1.0"
    static let bridgeSchema = "autoanim.review-bridge/1.0"
    static let maximumDocumentBytes = 8 * 1_024 * 1_024
    static let maximumArtifacts = 256
    static let maximumArtifactBytes: Int64 = 16 * 1_024 * 1_024 * 1_024
    static let maximumFrames = 1_800

    static let limitations = [
        "manifest_hmac_and_input_bytes_must_be_preverified_by_service_boundary",
        "manifest_hmac_signature_not_verified_by_review_bundle_core",
        "motion_consumption_reported_not_independently_recomputed",
        "closeup_bounds_require_native_semantic_selection",
        "material_hash_references_may_not_have_isolated_bytes",
        "raw_gnm_controls_not_sampled_by_a_measurement_viewport",
        "correction_writer_and_human_approval_not_established",
    ]

    static let protectedAnchorClasses = ["contact", "blink", "apex"]
    static let correctionReasonCodes = [
        "NATIVE_CORRECTION_WRITER_NOT_IMPLEMENTED",
        "CORRECTION_BRIDGE_MESSAGE_NOT_ENABLED",
        "HUMAN_REVIEW_NOT_RECORDED",
        "PRODUCTION_QUALIFICATION_NOT_ESTABLISHED",
    ]
}

enum ReviewValidation {
    static let jobAlphabet = Set("0123456789abcdefghjkmnpqrstvwxyz")
    static let identifierAlphabet = Set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:@+-"
    )
    static let lowercaseHex = Set("0123456789abcdef")
    static let bridgeRevisionAlphabet = Set("0123456789abcdefghijklmnopqrstuvwxyz._:-")

    static func isJobID(_ value: String) -> Bool {
        value.count == 26 && value.allSatisfy(jobAlphabet.contains)
    }

    static func isIdentifier(_ value: String) -> Bool {
        (1...200).contains(value.count) && value.allSatisfy(identifierAlphabet.contains)
    }

    static func isSHA256(_ value: String) -> Bool {
        value.count == 64 && value.allSatisfy(lowercaseHex.contains)
    }

    static func isRevisionID(_ value: String) -> Bool {
        let prefix = "review-revision:"
        return value.hasPrefix(prefix)
            && isSHA256(String(value.dropFirst(prefix.count)))
    }

    static func isBridgeRevisionID(_ value: String) -> Bool {
        guard (1...128).contains(value.count),
              value.first?.isASCII == true,
              value.first?.isLetter == true || value.first?.isNumber == true,
              value.last?.isASCII == true,
              value.last?.isLetter == true || value.last?.isNumber == true,
              !value.contains("..")
        else {
            return false
        }
        return value.allSatisfy(bridgeRevisionAlphabet.contains)
    }

    static func isLeafName(_ value: String) -> Bool {
        !value.isEmpty
            && value.utf8.count <= 255
            && value != "."
            && value != ".."
            && !value.contains("/")
            && !value.contains("\\")
    }

    static func gcd(_ lhs: UInt64, _ rhs: UInt64) -> UInt64 {
        var left = lhs
        var right = rhs
        while right != 0 {
            (left, right) = (right, left % right)
        }
        return left
    }
}

enum ReviewDigest {
    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func canonicalJSON(_ object: Any) throws -> Data {
        guard JSONSerialization.isValidJSONObject(object) else {
            throw ReviewModelError.invalid(
                field: "document",
                reason: "value is not finite JSON"
            )
        }
        return try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys, .withoutEscapingSlashes]
        )
    }

    static func canonicalJSON<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
    }

    static func sourcePTS(_ values: [Int64]) throws -> String {
        var bytes = Data("autoanim.review-source-pts/1.0".utf8)
        bytes.append(0)
        bytes.append(
            try canonicalJSON([
                "dtype": "<i8",
                "shape": [values.count],
            ] as [String: Any])
        )
        bytes.append(0)
        for value in values {
            var littleEndian = value.littleEndian
            withUnsafeBytes(of: &littleEndian) { bytes.append(contentsOf: $0) }
        }
        return sha256(bytes)
    }
}

enum ReviewJSONShape {
    static func object(
        _ value: Any,
        field: String,
        keys: Set<String>
    ) throws -> [String: Any] {
        guard let value = value as? [String: Any] else {
            throw ReviewModelError.invalid(field: field, reason: "must be an object")
        }
        try reviewRequire(
            Set(value.keys) == keys,
            field: field,
            reason: "fields differ from the fixed ReviewBundle v1 contract"
        )
        return value
    }

    static func array(_ value: Any, field: String) throws -> [Any] {
        guard let value = value as? [Any] else {
            throw ReviewModelError.invalid(field: field, reason: "must be an array")
        }
        return value
    }

    static func validate(_ root: [String: Any]) throws {
        let source = try object(
            root["source_manifest"] as Any,
            field: "source_manifest",
            keys: ["schema_version", "job_id", "kind", "status", "performance_manifest_sha256", "manifest_seal", "input"]
        )
        _ = try object(source["manifest_seal"] as Any, field: "source_manifest.manifest_seal", keys: ["schema", "key_id", "signature", "signature_verified"])
        _ = try object(source["input"] as Any, field: "source_manifest.input", keys: ["name", "sha256", "bytes", "media_type", "bytes_verified"])
        _ = try object(
            root["clock"] as Any,
            field: "clock",
            keys: [
                "schema_version", "capture_schema_version", "cursor_unit", "time_base",
                "display_time_mapping", "source_pts", "frame_count", "first_source_pts",
                "last_source_pts", "first_display_time_exact_rational",
                "source_start_time_exact_rational", "duration_exact_rational", "clock_sha256",
            ]
        )
        _ = try object(
            root["comparison_key"] as Any,
            field: "comparison_key",
            keys: [
                "schema_version", "input_sha256", "clock_sha256", "source_pts_sha256",
                "viewer_media_sha256", "gnm_version", "controls_performance_schema_version",
                "controls_identity_sha256", "comparison_key_sha256",
            ]
        )
        for (index, raw) in try array(root["artifacts"] as Any, field: "artifacts").enumerated() {
            _ = try object(raw, field: "artifacts.\(index)", keys: ["logical_name", "name", "bytes", "sha256", "media_type", "bytes_verified"])
        }
        for (index, raw) in try array(root["layers"] as Any, field: "layers").enumerated() {
            let layer = try object(
                raw,
                field: "layers.\(index)",
                keys: [
                    "schema_version", "layer_id", "layer_version", "revision_id",
                    "parent_revision_ids", "availability", "artifact_logical_names",
                    "motion_authority", "production_motion_authority", "consumption",
                    "changes_motion_reported", "production_validated", "approval_status",
                ]
            )
            _ = try object(layer["consumption"] as Any, field: "layers.\(index).consumption", keys: ["consumed_by_final_reported", "consumption_independently_verified"])
        }
        let graph = try object(
            root["revision_graph"] as Any,
            field: "revision_graph",
            keys: ["schema_version", "nodes", "edges", "ab_pairs", "ab_scope", "renderable_revisions", "immutable", "undo_redo_mode", "production_validated"]
        )
        for (index, raw) in try array(graph["nodes"] as Any, field: "revision_graph.nodes").enumerated() {
            _ = try object(raw, field: "revision_graph.nodes.\(index)", keys: ["revision_id", "layer_id", "parent_revision_ids", "immutable", "production_validated", "approval_status"])
        }
        for (index, raw) in try array(graph["edges"] as Any, field: "revision_graph.edges").enumerated() {
            _ = try object(raw, field: "revision_graph.edges.\(index)", keys: ["from_revision_id", "to_revision_id", "relation"])
        }
        for (index, raw) in try array(graph["renderable_revisions"] as Any, field: "revision_graph.renderable_revisions").enumerated() {
            _ = try object(raw, field: "revision_graph.renderable_revisions.\(index)", keys: ["revision_id", "artifact_logical_name", "render_role", "production_validated", "approval_status"])
        }
        for (index, raw) in try array(root["closeups"] as Any, field: "closeups").enumerated() {
            _ = try object(raw, field: "closeups.\(index)", keys: ["schema_version", "region_id", "region_version", "selection_space", "normalized_bounds", "selection_status", "renderable", "artifact_logical_names", "production_validated", "approval_status"])
        }
        for (index, raw) in try array(root["material_channels"] as Any, field: "material_channels").enumerated() {
            _ = try object(raw, field: "material_channels.\(index)", keys: ["schema_version", "channel", "manifest_key", "color_space", "status", "artifact_logical_name", "sha256", "isolatable", "measured", "production_validated", "approval_status"])
        }
        _ = try object(
            root["correction_eligibility"] as Any,
            field: "correction_eligibility",
            keys: [
                "schema_version", "candidate_request_eligible", "candidate_layer_id",
                "required_parent_revision_id", "selection_requires_exact_source_pts",
                "immutable_revision_required", "undo_redo_mode", "protected_anchor_classes",
                "writer_implemented", "production_revision_eligible", "human_review_recorded",
                "approval_status", "production_validated", "reason_codes",
            ]
        )
        _ = try object(root["bridge"] as Any, field: "bridge", keys: ["schema_version", "allowed_message_types", "message_version_required", "arbitrary_script_messages_allowed", "production_commands_enabled"])
        _ = try object(
            root["claims"] as Any,
            field: "claims",
            keys: [
                "artifact_ledger_bytes_verified", "exact_rational_pts_clock_verified",
                "manifest_signature_verified", "motion_consumption_independently_verified",
                "materials_approved", "correction_approved", "performance_approved",
                "production_validated", "publishable",
            ]
        )
    }
}

public struct ReviewRational: Codable, Equatable, Sendable {
    public let numerator: Int64
    public let denominator: Int64

    public init(_ numerator: Int64, _ denominator: Int64) {
        self.numerator = numerator
        self.denominator = denominator
    }

    public init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        numerator = try container.decode(Int64.self)
        denominator = try container.decode(Int64.self)
        guard container.isAtEnd else {
            throw ReviewModelError.invalid(
                field: decoder.codingPath.map(\.stringValue).joined(separator: "."),
                reason: "rational must contain exactly two integers"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        try container.encode(numerator)
        try container.encode(denominator)
    }

    func validate(field: String, positive: Bool = false, nonnegative: Bool = false) throws {
        try reviewRequire(denominator > 0, field: field, reason: "denominator must be positive")
        if positive {
            try reviewRequire(numerator > 0, field: field, reason: "numerator must be positive")
        } else if nonnegative {
            try reviewRequire(numerator >= 0, field: field, reason: "numerator must be nonnegative")
        }
        let divisor = ReviewValidation.gcd(numerator.magnitude, denominator.magnitude)
        try reviewRequire(divisor == 1, field: field, reason: "rational must be reduced")
    }

    static func product(_ integer: Int64, _ rational: ReviewRational, field: String) throws -> ReviewRational {
        let product = integer.multipliedReportingOverflow(by: rational.numerator)
        try reviewRequire(!product.overflow, field: field, reason: "rational multiplication overflowed")
        let divisor = ReviewValidation.gcd(product.partialValue.magnitude, rational.denominator.magnitude)
        return ReviewRational(
            product.partialValue / Int64(divisor),
            rational.denominator / Int64(divisor)
        )
    }
}

public struct ReviewSourceManifest: Codable, Equatable, Sendable {
    public let schemaVersion: String
    public let jobID: String
    public let kind: String
    public let status: String
    public let performanceManifestSHA256: String
    public let manifestSeal: ReviewManifestSeal
    public let input: ReviewSourceInput

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case jobID = "job_id"
        case kind
        case status
        case performanceManifestSHA256 = "performance_manifest_sha256"
        case manifestSeal = "manifest_seal"
        case input
    }
}

public struct ReviewManifestSeal: Codable, Equatable, Sendable {
    public let schema: String
    public let keyID: String
    public let signature: String
    public let signatureVerified: Bool

    enum CodingKeys: String, CodingKey {
        case schema
        case keyID = "key_id"
        case signature
        case signatureVerified = "signature_verified"
    }
}

public struct ReviewSourceInput: Codable, Equatable, Sendable {
    public let name: String
    public let sha256: String
    public let bytes: Int64
    public let mediaType: String
    public let bytesVerified: Bool

    enum CodingKeys: String, CodingKey {
        case name
        case sha256
        case bytes
        case mediaType = "media_type"
        case bytesVerified = "bytes_verified"
    }
}

public struct ReviewClock: Codable, Equatable, Sendable {
    public let schemaVersion: String
    public let captureSchemaVersion: String
    public let cursorUnit: String
    public let timeBase: ReviewRational
    public let displayTimeMapping: String
    public let sourcePTS: [Int64]
    public let frameCount: Int
    public let firstSourcePTS: Int64
    public let lastSourcePTS: Int64
    public let firstDisplayTimeExactRational: ReviewRational
    public let sourceStartTimeExactRational: ReviewRational
    public let durationExactRational: ReviewRational
    public let clockSHA256: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case captureSchemaVersion = "capture_schema_version"
        case cursorUnit = "cursor_unit"
        case timeBase = "time_base"
        case displayTimeMapping = "display_time_mapping"
        case sourcePTS = "source_pts"
        case frameCount = "frame_count"
        case firstSourcePTS = "first_source_pts"
        case lastSourcePTS = "last_source_pts"
        case firstDisplayTimeExactRational = "first_display_time_exact_rational"
        case sourceStartTimeExactRational = "source_start_time_exact_rational"
        case durationExactRational = "duration_exact_rational"
        case clockSHA256 = "clock_sha256"
    }

    func hashPayload() -> [String: Any] {
        [
            "schema_version": schemaVersion,
            "capture_schema_version": captureSchemaVersion,
            "cursor_unit": cursorUnit,
            "time_base": [timeBase.numerator, timeBase.denominator],
            "display_time_mapping": displayTimeMapping,
            "source_pts": sourcePTS,
            "frame_count": frameCount,
            "first_source_pts": firstSourcePTS,
            "last_source_pts": lastSourcePTS,
            "first_display_time_exact_rational": [
                firstDisplayTimeExactRational.numerator,
                firstDisplayTimeExactRational.denominator,
            ],
            "source_start_time_exact_rational": [
                sourceStartTimeExactRational.numerator,
                sourceStartTimeExactRational.denominator,
            ],
            "duration_exact_rational": [
                durationExactRational.numerator,
                durationExactRational.denominator,
            ],
        ]
    }
}

public struct ReviewComparisonKey: Codable, Equatable, Sendable {
    public let schemaVersion: String
    public let inputSHA256: String
    public let clockSHA256: String
    public let sourcePTSSHA256: String
    public let viewerMediaSHA256: String
    public let gnmVersion: String
    public let controlsPerformanceSchemaVersion: String
    public let controlsIdentitySHA256: String
    public let comparisonKeySHA256: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case inputSHA256 = "input_sha256"
        case clockSHA256 = "clock_sha256"
        case sourcePTSSHA256 = "source_pts_sha256"
        case viewerMediaSHA256 = "viewer_media_sha256"
        case gnmVersion = "gnm_version"
        case controlsPerformanceSchemaVersion = "controls_performance_schema_version"
        case controlsIdentitySHA256 = "controls_identity_sha256"
        case comparisonKeySHA256 = "comparison_key_sha256"
    }

    func hashPayload() -> [String: Any] {
        [
            "schema_version": schemaVersion,
            "input_sha256": inputSHA256,
            "clock_sha256": clockSHA256,
            "source_pts_sha256": sourcePTSSHA256,
            "viewer_media_sha256": viewerMediaSHA256,
            "gnm_version": gnmVersion,
            "controls_performance_schema_version": controlsPerformanceSchemaVersion,
            "controls_identity_sha256": controlsIdentitySHA256,
        ]
    }
}

public struct ReviewArtifact: Codable, Equatable, Sendable {
    public let logicalName: String
    public let name: String
    public let bytes: Int64
    public let sha256: String
    public let mediaType: String
    public let bytesVerified: Bool

    enum CodingKeys: String, CodingKey {
        case logicalName = "logical_name"
        case name
        case bytes
        case sha256
        case mediaType = "media_type"
        case bytesVerified = "bytes_verified"
    }
}

public enum ReviewLayerID: String, Codable, CaseIterable, Sendable {
    case source
    case visualBase = "visual_base"
    case audioRepair = "audio_repair"
    case acting
    case authoredCorrection = "authored_correction"
    case physics
    case final
}

public struct ReviewLayerConsumption: Codable, Equatable, Sendable {
    public let consumedByFinalReported: Bool
    public let consumptionIndependentlyVerified: Bool

    enum CodingKeys: String, CodingKey {
        case consumedByFinalReported = "consumed_by_final_reported"
        case consumptionIndependentlyVerified = "consumption_independently_verified"
    }
}

public struct ReviewLayer: Codable, Equatable, Sendable {
    public let schemaVersion: String
    public let layerID: ReviewLayerID
    public let layerVersion: Int
    public let revisionID: String
    public let parentRevisionIDs: [String]
    public let availability: String
    public let artifactLogicalNames: [String]
    public let motionAuthority: String
    public let productionMotionAuthority: String
    public let consumption: ReviewLayerConsumption
    public let changesMotionReported: Bool
    public let productionValidated: Bool
    public let approvalStatus: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case layerID = "layer_id"
        case layerVersion = "layer_version"
        case revisionID = "revision_id"
        case parentRevisionIDs = "parent_revision_ids"
        case availability
        case artifactLogicalNames = "artifact_logical_names"
        case motionAuthority = "motion_authority"
        case productionMotionAuthority = "production_motion_authority"
        case consumption
        case changesMotionReported = "changes_motion_reported"
        case productionValidated = "production_validated"
        case approvalStatus = "approval_status"
    }

    func revisionPayload() -> ReviewLayerRevisionPayload {
        ReviewLayerRevisionPayload(
            schemaVersion: schemaVersion,
            layerID: layerID,
            layerVersion: layerVersion,
            parentRevisionIDs: parentRevisionIDs,
            availability: availability,
            artifactLogicalNames: artifactLogicalNames,
            motionAuthority: motionAuthority,
            productionMotionAuthority: productionMotionAuthority,
            consumption: consumption,
            changesMotionReported: changesMotionReported,
            productionValidated: productionValidated,
            approvalStatus: approvalStatus
        )
    }
}

struct ReviewLayerRevisionPayload: Codable {
    let schemaVersion: String
    let layerID: ReviewLayerID
    let layerVersion: Int
    let parentRevisionIDs: [String]
    let availability: String
    let artifactLogicalNames: [String]
    let motionAuthority: String
    let productionMotionAuthority: String
    let consumption: ReviewLayerConsumption
    let changesMotionReported: Bool
    let productionValidated: Bool
    let approvalStatus: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case layerID = "layer_id"
        case layerVersion = "layer_version"
        case parentRevisionIDs = "parent_revision_ids"
        case availability
        case artifactLogicalNames = "artifact_logical_names"
        case motionAuthority = "motion_authority"
        case productionMotionAuthority = "production_motion_authority"
        case consumption
        case changesMotionReported = "changes_motion_reported"
        case productionValidated = "production_validated"
        case approvalStatus = "approval_status"
    }
}

public struct ReviewRevisionGraph: Codable, Equatable, Sendable {
    public let schemaVersion: String
    public let nodes: [ReviewRevisionNode]
    public let edges: [ReviewRevisionEdge]
    public let abPairs: [ReviewABPair]
    public let abScope: String
    public let renderableRevisions: [ReviewRenderableRevision]
    public let immutable: Bool
    public let undoRedoMode: String
    public let productionValidated: Bool

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case nodes
        case edges
        case abPairs = "ab_pairs"
        case abScope = "ab_scope"
        case renderableRevisions = "renderable_revisions"
        case immutable
        case undoRedoMode = "undo_redo_mode"
        case productionValidated = "production_validated"
    }
}

public struct ReviewRevisionNode: Codable, Equatable, Sendable {
    public let revisionID: String
    public let layerID: ReviewLayerID
    public let parentRevisionIDs: [String]
    public let immutable: Bool
    public let productionValidated: Bool
    public let approvalStatus: String

    enum CodingKeys: String, CodingKey {
        case revisionID = "revision_id"
        case layerID = "layer_id"
        case parentRevisionIDs = "parent_revision_ids"
        case immutable
        case productionValidated = "production_validated"
        case approvalStatus = "approval_status"
    }
}

public struct ReviewRevisionEdge: Codable, Equatable, Sendable {
    public let fromRevisionID: String
    public let toRevisionID: String
    public let relation: String

    enum CodingKeys: String, CodingKey {
        case fromRevisionID = "from_revision_id"
        case toRevisionID = "to_revision_id"
        case relation
    }
}

public struct ReviewABPair: Codable, Equatable, Sendable {
    public let aRevisionID: String
    public let bRevisionID: String

    enum CodingKeys: String, CodingKey {
        case aRevisionID = "a_revision_id"
        case bRevisionID = "b_revision_id"
    }
}

public struct ReviewRenderableRevision: Codable, Equatable, Sendable {
    public let revisionID: String
    public let artifactLogicalName: String
    public let renderRole: String
    public let productionValidated: Bool
    public let approvalStatus: String

    enum CodingKeys: String, CodingKey {
        case revisionID = "revision_id"
        case artifactLogicalName = "artifact_logical_name"
        case renderRole = "render_role"
        case productionValidated = "production_validated"
        case approvalStatus = "approval_status"
    }
}

public struct ReviewCloseup: Codable, Equatable, Sendable {
    public let schemaVersion: String
    public let regionID: String
    public let regionVersion: Int
    public let selectionSpace: String
    public let normalizedBounds: [Double]?
    public let selectionStatus: String
    public let renderable: Bool
    public let artifactLogicalNames: [String]
    public let productionValidated: Bool
    public let approvalStatus: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case regionID = "region_id"
        case regionVersion = "region_version"
        case selectionSpace = "selection_space"
        case normalizedBounds = "normalized_bounds"
        case selectionStatus = "selection_status"
        case renderable
        case artifactLogicalNames = "artifact_logical_names"
        case productionValidated = "production_validated"
        case approvalStatus = "approval_status"
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(regionID, forKey: .regionID)
        try container.encode(regionVersion, forKey: .regionVersion)
        try container.encode(selectionSpace, forKey: .selectionSpace)
        if let normalizedBounds {
            try container.encode(normalizedBounds, forKey: .normalizedBounds)
        } else {
            try container.encodeNil(forKey: .normalizedBounds)
        }
        try container.encode(selectionStatus, forKey: .selectionStatus)
        try container.encode(renderable, forKey: .renderable)
        try container.encode(artifactLogicalNames, forKey: .artifactLogicalNames)
        try container.encode(productionValidated, forKey: .productionValidated)
        try container.encode(approvalStatus, forKey: .approvalStatus)
    }
}

public struct ReviewMaterialChannel: Codable, Equatable, Sendable {
    public let schemaVersion: String
    public let channel: String
    public let manifestKey: String
    public let colorSpace: String
    public let status: String
    public let artifactLogicalName: String?
    public let sha256: String?
    public let isolatable: Bool
    public let measured: Bool
    public let productionValidated: Bool
    public let approvalStatus: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case channel
        case manifestKey = "manifest_key"
        case colorSpace = "color_space"
        case status
        case artifactLogicalName = "artifact_logical_name"
        case sha256
        case isolatable
        case measured
        case productionValidated = "production_validated"
        case approvalStatus = "approval_status"
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(channel, forKey: .channel)
        try container.encode(manifestKey, forKey: .manifestKey)
        try container.encode(colorSpace, forKey: .colorSpace)
        try container.encode(status, forKey: .status)
        if let artifactLogicalName {
            try container.encode(artifactLogicalName, forKey: .artifactLogicalName)
        } else {
            try container.encodeNil(forKey: .artifactLogicalName)
        }
        if let sha256 {
            try container.encode(sha256, forKey: .sha256)
        } else {
            try container.encodeNil(forKey: .sha256)
        }
        try container.encode(isolatable, forKey: .isolatable)
        try container.encode(measured, forKey: .measured)
        try container.encode(productionValidated, forKey: .productionValidated)
        try container.encode(approvalStatus, forKey: .approvalStatus)
    }
}

public struct ReviewCorrectionEligibility: Codable, Equatable, Sendable {
    public let schemaVersion: String
    public let candidateRequestEligible: Bool
    public let candidateLayerID: ReviewLayerID
    public let requiredParentRevisionID: String
    public let selectionRequiresExactSourcePTS: Bool
    public let immutableRevisionRequired: Bool
    public let undoRedoMode: String
    public let protectedAnchorClasses: [String]
    public let writerImplemented: Bool
    public let productionRevisionEligible: Bool
    public let humanReviewRecorded: Bool
    public let approvalStatus: String
    public let productionValidated: Bool
    public let reasonCodes: [String]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case candidateRequestEligible = "candidate_request_eligible"
        case candidateLayerID = "candidate_layer_id"
        case requiredParentRevisionID = "required_parent_revision_id"
        case selectionRequiresExactSourcePTS = "selection_requires_exact_source_pts"
        case immutableRevisionRequired = "immutable_revision_required"
        case undoRedoMode = "undo_redo_mode"
        case protectedAnchorClasses = "protected_anchor_classes"
        case writerImplemented = "writer_implemented"
        case productionRevisionEligible = "production_revision_eligible"
        case humanReviewRecorded = "human_review_recorded"
        case approvalStatus = "approval_status"
        case productionValidated = "production_validated"
        case reasonCodes = "reason_codes"
    }
}

public struct ReviewBridgeContract: Codable, Equatable, Sendable {
    public let schemaVersion: String
    public let allowedMessageTypes: [String]
    public let messageVersionRequired: Bool
    public let arbitraryScriptMessagesAllowed: Bool
    public let productionCommandsEnabled: Bool

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case allowedMessageTypes = "allowed_message_types"
        case messageVersionRequired = "message_version_required"
        case arbitraryScriptMessagesAllowed = "arbitrary_script_messages_allowed"
        case productionCommandsEnabled = "production_commands_enabled"
    }
}

public struct ReviewClaims: Codable, Equatable, Sendable {
    public let artifactLedgerBytesVerified: Bool
    public let exactRationalPTSClockVerified: Bool
    public let manifestSignatureVerified: Bool
    public let motionConsumptionIndependentlyVerified: Bool
    public let materialsApproved: Bool
    public let correctionApproved: Bool
    public let performanceApproved: Bool
    public let productionValidated: Bool
    public let publishable: Bool

    enum CodingKeys: String, CodingKey {
        case artifactLedgerBytesVerified = "artifact_ledger_bytes_verified"
        case exactRationalPTSClockVerified = "exact_rational_pts_clock_verified"
        case manifestSignatureVerified = "manifest_signature_verified"
        case motionConsumptionIndependentlyVerified = "motion_consumption_independently_verified"
        case materialsApproved = "materials_approved"
        case correctionApproved = "correction_approved"
        case performanceApproved = "performance_approved"
        case productionValidated = "production_validated"
        case publishable
    }
}

public struct ReviewBundle: Codable, Equatable, Sendable {
    public let schemaVersion: String
    public let sourceManifest: ReviewSourceManifest
    public let clock: ReviewClock
    public let comparisonKey: ReviewComparisonKey
    public let artifacts: [ReviewArtifact]
    public let layers: [ReviewLayer]
    public let revisionGraph: ReviewRevisionGraph
    public let closeups: [ReviewCloseup]
    public let materialChannels: [ReviewMaterialChannel]
    public let correctionEligibility: ReviewCorrectionEligibility
    public let bridge: ReviewBridgeContract
    public let claims: ReviewClaims
    public let limitations: [String]
    public let bundleSHA256: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case sourceManifest = "source_manifest"
        case clock
        case comparisonKey = "comparison_key"
        case artifacts
        case layers
        case revisionGraph = "revision_graph"
        case closeups
        case materialChannels = "material_channels"
        case correctionEligibility = "correction_eligibility"
        case bridge
        case claims
        case limitations
        case bundleSHA256 = "bundle_sha256"
    }

    public static func decodeStrict(from data: Data) throws -> ReviewBundle {
        try reviewRequire(
            !data.isEmpty && data.count <= ReviewContract.maximumDocumentBytes,
            field: "review_bundle",
            reason: "document is empty or exceeds 8 MiB"
        )
        let raw = try JSONSerialization.jsonObject(with: data)
        guard var root = raw as? [String: Any] else {
            throw ReviewModelError.invalid(field: "review_bundle", reason: "root must be an object")
        }
        let canonicalDocument = try ReviewDigest.canonicalJSON(root)
        try reviewRequire(
            data == canonicalDocument,
            field: "review_bundle",
            reason: "document must be canonical JSON without duplicate keys"
        )
        let expectedRootKeys = Set(CodingKeys.allCases.map(\.rawValue))
        try reviewRequire(
            Set(root.keys) == expectedRootKeys,
            field: "review_bundle",
            reason: "top-level fields differ from ReviewBundle v1"
        )
        try ReviewJSONShape.validate(root)
        guard let declaredBundleHash = root.removeValue(forKey: "bundle_sha256") as? String else {
            throw ReviewModelError.invalid(field: "bundle_sha256", reason: "hash is missing")
        }
        try reviewRequire(
            ReviewValidation.isSHA256(declaredBundleHash),
            field: "bundle_sha256",
            reason: "must be lowercase SHA-256"
        )
        let calculatedBundleHash = ReviewDigest.sha256(try ReviewDigest.canonicalJSON(root))
        try reviewRequire(
            calculatedBundleHash == declaredBundleHash,
            field: "bundle_sha256",
            reason: "payload hash does not match"
        )
        let bundle = try JSONDecoder().decode(ReviewBundle.self, from: data)
        try bundle.validate()
        return bundle
    }

    public func validate() throws {
        try reviewRequire(
            schemaVersion == ReviewContract.bundleSchema,
            field: "schema_version",
            reason: "unsupported schema"
        )
        try validateSource()
        try validateClock()
        let artifactsByName = try validateArtifacts()
        try validateComparison(artifactsByName: artifactsByName)
        try validateLayers(artifactsByName: artifactsByName)
        try validateRevisionGraph()
        try validateCloseups()
        try validateMaterials(artifactsByName: artifactsByName)
        try validateCorrection()
        try validateFailClosedClaims()
        try reviewRequire(
            ReviewValidation.isSHA256(bundleSHA256),
            field: "bundle_sha256",
            reason: "must be lowercase SHA-256"
        )
    }

    private func validateSource() throws {
        let source = sourceManifest
        try reviewRequire(source.schemaVersion == "1.0", field: "source_manifest.schema_version", reason: "must be 1.0")
        try reviewRequire(ReviewValidation.isJobID(source.jobID), field: "source_manifest.job_id", reason: "must be a canonical 26-character AutoAnim job ID")
        try reviewRequire(source.kind == "video_performance", field: "source_manifest.kind", reason: "must be video_performance")
        try reviewRequire(source.status == "succeeded", field: "source_manifest.status", reason: "must be succeeded")
        try reviewRequire(ReviewValidation.isSHA256(source.performanceManifestSHA256), field: "source_manifest.performance_manifest_sha256", reason: "must be lowercase SHA-256")
        try reviewRequire(source.manifestSeal.schema == "autoanim.hmac-sha256.v1", field: "source_manifest.manifest_seal.schema", reason: "unsupported seal")
        try reviewRequire(!source.manifestSeal.keyID.isEmpty && source.manifestSeal.keyID.utf8.count <= 64, field: "source_manifest.manifest_seal.key_id", reason: "must be bounded text")
        try reviewRequire(ReviewValidation.isSHA256(source.manifestSeal.signature), field: "source_manifest.manifest_seal.signature", reason: "must be lowercase SHA-256")
        try reviewRequire(!source.manifestSeal.signatureVerified, field: "source_manifest.manifest_seal.signature_verified", reason: "core cannot claim signature verification")
        try reviewRequire(ReviewValidation.isLeafName(source.input.name), field: "source_manifest.input.name", reason: "must be a leaf filename")
        try reviewRequire(ReviewValidation.isSHA256(source.input.sha256), field: "source_manifest.input.sha256", reason: "must be lowercase SHA-256")
        try reviewRequire((1...ReviewContract.maximumArtifactBytes).contains(source.input.bytes), field: "source_manifest.input.bytes", reason: "must be within the v1 byte bound")
        try reviewRequire(!source.input.mediaType.isEmpty && source.input.mediaType.utf8.count <= 160, field: "source_manifest.input.media_type", reason: "must be bounded text")
        try reviewRequire(!source.input.bytesVerified, field: "source_manifest.input.bytes_verified", reason: "input bytes are not verified by bundle core")
    }

    private func validateClock() throws {
        try reviewRequire(clock.schemaVersion == ReviewContract.clockSchema, field: "clock.schema_version", reason: "unsupported schema")
        try reviewRequire(clock.captureSchemaVersion == "autoanim.capture.v1", field: "clock.capture_schema_version", reason: "must be Capture v1")
        try reviewRequire(clock.cursorUnit == "source_pts", field: "clock.cursor_unit", reason: "must be source_pts")
        try reviewRequire(clock.displayTimeMapping == "(source_pts-first_source_pts)*time_base", field: "clock.display_time_mapping", reason: "unsupported source-to-display mapping")
        try clock.timeBase.validate(field: "clock.time_base", positive: true)
        try reviewRequire((1...ReviewContract.maximumFrames).contains(clock.sourcePTS.count), field: "clock.source_pts", reason: "must contain 1...1800 frames")
        try reviewRequire(clock.frameCount == clock.sourcePTS.count, field: "clock.frame_count", reason: "does not match source PTS")
        try reviewRequire(clock.sourcePTS.first == clock.firstSourcePTS, field: "clock.first_source_pts", reason: "does not match source PTS")
        try reviewRequire(clock.sourcePTS.last == clock.lastSourcePTS, field: "clock.last_source_pts", reason: "does not match source PTS")
        for (left, right) in zip(clock.sourcePTS, clock.sourcePTS.dropFirst()) {
            try reviewRequire(right > left, field: "clock.source_pts", reason: "must be strictly increasing")
        }
        try clock.firstDisplayTimeExactRational.validate(field: "clock.first_display_time_exact_rational")
        try clock.sourceStartTimeExactRational.validate(field: "clock.source_start_time_exact_rational")
        try clock.durationExactRational.validate(field: "clock.duration_exact_rational", nonnegative: true)
        let expectedSourceStart = try ReviewRational.product(clock.firstSourcePTS, clock.timeBase, field: "clock.source_start_time_exact_rational")
        let delta = clock.lastSourcePTS.subtractingReportingOverflow(clock.firstSourcePTS)
        try reviewRequire(!delta.overflow, field: "clock.duration_exact_rational", reason: "PTS duration overflowed")
        let expectedDuration = try ReviewRational.product(delta.partialValue, clock.timeBase, field: "clock.duration_exact_rational")
        try reviewRequire(clock.firstDisplayTimeExactRational == ReviewRational(0, 1), field: "clock.first_display_time_exact_rational", reason: "display cursor must start at zero")
        try reviewRequire(clock.sourceStartTimeExactRational == expectedSourceStart, field: "clock.source_start_time_exact_rational", reason: "does not equal first PTS times time base")
        try reviewRequire(clock.durationExactRational == expectedDuration, field: "clock.duration_exact_rational", reason: "does not equal PTS duration")
        try reviewRequire(ReviewValidation.isSHA256(clock.clockSHA256), field: "clock.clock_sha256", reason: "must be lowercase SHA-256")
        let calculated = ReviewDigest.sha256(try ReviewDigest.canonicalJSON(clock.hashPayload()))
        try reviewRequire(calculated == clock.clockSHA256, field: "clock.clock_sha256", reason: "clock payload hash does not match")
    }

    private func validateArtifacts() throws -> [String: ReviewArtifact] {
        try reviewRequire((1...ReviewContract.maximumArtifacts).contains(artifacts.count), field: "artifacts", reason: "artifact count is outside bounds")
        let names = artifacts.map(\.logicalName)
        try reviewRequire(names == names.sorted(), field: "artifacts", reason: "must be sorted by logical name")
        try reviewRequire(Set(names).count == names.count, field: "artifacts", reason: "logical names must be unique")
        var result = [String: ReviewArtifact]()
        for (index, artifact) in artifacts.enumerated() {
            let field = "artifacts.\(index)"
            try reviewRequire(ReviewValidation.isIdentifier(artifact.logicalName), field: "\(field).logical_name", reason: "identifier is invalid")
            try reviewRequire(ReviewValidation.isLeafName(artifact.name), field: "\(field).name", reason: "must be a leaf filename")
            try reviewRequire((1...ReviewContract.maximumArtifactBytes).contains(artifact.bytes), field: "\(field).bytes", reason: "size is outside bounds")
            try reviewRequire(ReviewValidation.isSHA256(artifact.sha256), field: "\(field).sha256", reason: "must be lowercase SHA-256")
            try reviewRequire(!artifact.mediaType.isEmpty && artifact.mediaType.utf8.count <= 160, field: "\(field).media_type", reason: "must be bounded text")
            try reviewRequire(artifact.bytesVerified, field: "\(field).bytes_verified", reason: "artifact bytes must be verified")
            result[artifact.logicalName] = artifact
        }
        try reviewRequire(Set(["capture", "controls", "viewer_media", "glb"]).isSubset(of: Set(names)), field: "artifacts", reason: "capture, controls, viewer_media, and glb are required")
        return result
    }

    private func validateComparison(artifactsByName: [String: ReviewArtifact]) throws {
        let key = comparisonKey
        try reviewRequire(key.schemaVersion == ReviewContract.comparisonSchema, field: "comparison_key.schema_version", reason: "unsupported schema")
        let hashFields = [
            ("input_sha256", key.inputSHA256),
            ("clock_sha256", key.clockSHA256),
            ("source_pts_sha256", key.sourcePTSSHA256),
            ("viewer_media_sha256", key.viewerMediaSHA256),
            ("controls_identity_sha256", key.controlsIdentitySHA256),
            ("comparison_key_sha256", key.comparisonKeySHA256),
        ]
        for (field, value) in hashFields {
            try reviewRequire(ReviewValidation.isSHA256(value), field: "comparison_key.\(field)", reason: "must be lowercase SHA-256")
        }
        try reviewRequire(key.inputSHA256 == sourceManifest.input.sha256, field: "comparison_key.input_sha256", reason: "does not match source input")
        try reviewRequire(key.clockSHA256 == clock.clockSHA256, field: "comparison_key.clock_sha256", reason: "does not match review clock")
        let calculatedSourcePTSSHA256 = try ReviewDigest.sourcePTS(clock.sourcePTS)
        try reviewRequire(key.sourcePTSSHA256 == calculatedSourcePTSSHA256, field: "comparison_key.source_pts_sha256", reason: "does not match exact source PTS")
        try reviewRequire(key.viewerMediaSHA256 == artifactsByName["viewer_media"]?.sha256, field: "comparison_key.viewer_media_sha256", reason: "does not match viewer proxy")
        try reviewRequire(key.gnmVersion == "3.0", field: "comparison_key.gnm_version", reason: "ReviewBundle v1 requires GNM 3.0")
        try reviewRequire(key.controlsPerformanceSchemaVersion == "autoanim.gnm-performance.v3", field: "comparison_key.controls_performance_schema_version", reason: "must be Performance v3")
        let calculated = ReviewDigest.sha256(try ReviewDigest.canonicalJSON(key.hashPayload()))
        try reviewRequire(calculated == key.comparisonKeySHA256, field: "comparison_key.comparison_key_sha256", reason: "comparison key payload hash does not match")
    }

    private func artifactLayer(_ name: String) -> ReviewLayerID? {
        let sourceNames: Set<String> = [
            "capture", "capture_jsonl", "performance_evidence", "pixel_observations",
            "observation_v3", "video_capture_run", "visual_track", "visual_track_summary",
            "capture_session", "viewer_media", "audio_video_timing",
            "audio_visual_timing_consumption", "audio_visual_source", "oral_validation",
            "oral_glb_validation", "performance_revision_chain",
        ]
        if sourceNames.contains(name) || name.hasPrefix("audio_visual_source_") { return .source }
        if ["controls", "controls_jsonl", "retarget_calibration"].contains(name) { return .visualBase }
        if ["audio_visual_repair", "audio_visual_repair_arrays"].contains(name) { return .audioRepair }
        if name.hasPrefix("acting_") || name.hasPrefix("direction_") { return .acting }
        if ["mouth_aperture_edit", "mouth_aperture_edit_arrays"].contains(name) { return .authoredCorrection }
        if name == "physics" || name.hasPrefix("physics_") { return .physics }
        if ["glb", "glb_mapping"].contains(name) { return .final }
        return nil
    }

    private func layerAvailable(_ id: ReviewLayerID, names: [String]) -> Bool {
        let values = Set(names)
        switch id {
        case .audioRepair:
            return Set(["audio_visual_repair", "audio_visual_repair_arrays"]).isSubset(of: values)
        case .authoredCorrection:
            return Set(["mouth_aperture_edit", "mouth_aperture_edit_arrays"]).isSubset(of: values)
        case .acting:
            return !values.intersection(["acting_track", "acting_applied_controls", "body_track", "body_track_arrays"]).isEmpty
        case .final:
            return values.contains("glb")
        default:
            return !values.isEmpty
        }
    }

    private func expectedMotionAuthority(_ id: ReviewLayerID, available: Bool, changesMotion: Bool) -> String {
        if id == .source { return "reference_only" }
        guard available && changesMotion else { return "none" }
        switch id {
        case .source: return "reference_only"
        case .visualBase: return "candidate_visual_retarget"
        case .audioRepair: return "candidate_lower_face_and_tongue_repair"
        case .acting: return "candidate_acting_override"
        case .authoredCorrection: return "candidate_bounded_artist_override"
        case .physics: return "candidate_simulation"
        case .final: return "candidate_composite"
        }
    }

    private func validateLayers(artifactsByName: [String: ReviewArtifact]) throws {
        let expectedIDs = ReviewLayerID.allCases
        try reviewRequire(layers.map(\.layerID) == expectedIDs, field: "layers", reason: "must contain the seven canonical layers in order")
        let finalAvailable = artifactsByName.keys.contains { artifactLayer($0) == .final }
        var previousRevision: String?
        for (index, layer) in layers.enumerated() {
            let field = "layers.\(index)"
            let expectedNames = artifactsByName.keys.filter { artifactLayer($0) == layer.layerID }.sorted()
            let available = layerAvailable(layer.layerID, names: expectedNames)
            let changesMotion = layer.changesMotionReported
            try reviewRequire(layer.schemaVersion == ReviewContract.layerSchema, field: "\(field).schema_version", reason: "unsupported schema")
            try reviewRequire(layer.layerVersion == 1, field: "\(field).layer_version", reason: "must be 1")
            try reviewRequire(ReviewValidation.isRevisionID(layer.revisionID), field: "\(field).revision_id", reason: "must be a review-revision SHA-256 identifier")
            try reviewRequire(layer.parentRevisionIDs == (previousRevision.map { [$0] } ?? []), field: "\(field).parent_revision_ids", reason: "must form one immutable chain")
            try reviewRequire(layer.availability == (available ? "available" : "unavailable"), field: "\(field).availability", reason: "does not match artifacts")
            try reviewRequire(layer.artifactLogicalNames == expectedNames, field: "\(field).artifact_logical_names", reason: "does not match artifact grouping")
            try reviewRequire(!( [.source, .acting, .physics].contains(layer.layerID) && changesMotion), field: "\(field).changes_motion_reported", reason: "layer cannot claim applied motion in ReviewBundle v1")
            try reviewRequire(!(!available && changesMotion), field: "\(field).changes_motion_reported", reason: "unavailable layer cannot change motion")
            try reviewRequire(!([.visualBase, .final].contains(layer.layerID) && available && !changesMotion), field: "\(field).changes_motion_reported", reason: "intrinsic visual and final revisions must report motion")
            try reviewRequire(layer.motionAuthority == expectedMotionAuthority(layer.layerID, available: available, changesMotion: changesMotion), field: "\(field).motion_authority", reason: "authority is inconsistent")
            try reviewRequire(layer.productionMotionAuthority == "none", field: "\(field).production_motion_authority", reason: "must remain fail closed")
            let consumed = finalAvailable && changesMotion && [.visualBase, .audioRepair, .acting, .authoredCorrection, .physics].contains(layer.layerID)
            try reviewRequire(layer.consumption.consumedByFinalReported == consumed, field: "\(field).consumption.consumed_by_final_reported", reason: "reported consumption is inconsistent")
            try reviewRequire(!layer.consumption.consumptionIndependentlyVerified, field: "\(field).consumption.consumption_independently_verified", reason: "must remain false")
            try reviewRequire(!layer.productionValidated && layer.approvalStatus == "unapproved", field: field, reason: "layer must remain unapproved and unvalidated")
            let expectedRevision = "review-revision:" + ReviewDigest.sha256(try ReviewDigest.canonicalJSON(layer.revisionPayload()))
            try reviewRequire(layer.revisionID == expectedRevision, field: "\(field).revision_id", reason: "revision payload hash does not match")
            previousRevision = layer.revisionID
        }
    }

    private func validateRevisionGraph() throws {
        let graph = revisionGraph
        try reviewRequire(graph.schemaVersion == ReviewContract.revisionGraphSchema, field: "revision_graph.schema_version", reason: "unsupported schema")
        let expectedNodes = layers.map {
            ReviewRevisionNode(
                revisionID: $0.revisionID,
                layerID: $0.layerID,
                parentRevisionIDs: $0.parentRevisionIDs,
                immutable: true,
                productionValidated: false,
                approvalStatus: "unapproved"
            )
        }
        let expectedEdges = layers.indices.dropFirst().map { index in
            ReviewRevisionEdge(
                fromRevisionID: layers[index - 1].revisionID,
                toRevisionID: layers[index].revisionID,
                relation: "candidate_layer_composition"
            )
        }
        try reviewRequire(graph.nodes == expectedNodes, field: "revision_graph.nodes", reason: "must exactly mirror the layer chain")
        try reviewRequire(graph.edges == expectedEdges, field: "revision_graph.edges", reason: "must exactly mirror the layer chain")
        try reviewRequire(graph.abPairs.isEmpty, field: "revision_graph.ab_pairs", reason: "within-job A/B is unsupported")
        try reviewRequire(graph.abScope == "cross_bundle_same_comparison_key_only", field: "revision_graph.ab_scope", reason: "must be cross-bundle only")
        try reviewRequire(graph.renderableRevisions.count == 1, field: "revision_graph.renderable_revisions", reason: "exactly one final revision must be renderable")
        if let renderable = graph.renderableRevisions.first, let final = layers.last {
            try reviewRequire(
                renderable == ReviewRenderableRevision(
                    revisionID: final.revisionID,
                    artifactLogicalName: "glb",
                    renderRole: "final_textured_animation",
                    productionValidated: false,
                    approvalStatus: "unapproved"
                ),
                field: "revision_graph.renderable_revisions.0",
                reason: "must be the unapproved final GLB revision"
            )
        }
        try reviewRequire(graph.immutable, field: "revision_graph.immutable", reason: "must be immutable")
        try reviewRequire(graph.undoRedoMode == "append_only_revision_selection", field: "revision_graph.undo_redo_mode", reason: "unsupported mode")
        try reviewRequire(!graph.productionValidated, field: "revision_graph.production_validated", reason: "must remain false")
    }

    private func validateCloseups() throws {
        let expectedRegions = ["mouth", "tongue", "left_eye", "right_eye"]
        try reviewRequire(closeups.map(\.regionID) == expectedRegions, field: "closeups", reason: "regions or order differ")
        for (index, closeup) in closeups.enumerated() {
            let field = "closeups.\(index)"
            try reviewRequire(closeup.schemaVersion == ReviewContract.closeupSchema, field: "\(field).schema_version", reason: "unsupported schema")
            try reviewRequire(closeup.regionVersion == 1 && closeup.selectionSpace == "semantic_face_region", field: field, reason: "unsupported region definition")
            try reviewRequire(closeup.normalizedBounds == nil && closeup.selectionStatus == "native_selection_required" && !closeup.renderable, field: field, reason: "closeup must remain non-renderable until native selection")
            try reviewRequire(closeup.artifactLogicalNames == ["viewer_media", "glb"], field: "\(field).artifact_logical_names", reason: "review sources differ")
            try reviewRequire(!closeup.productionValidated && closeup.approvalStatus == "unapproved", field: field, reason: "closeup must remain fail closed")
        }
    }

    private func validateMaterials(artifactsByName: [String: ReviewArtifact]) throws {
        let expected = [
            ("base_color", "base_color", "srgb"),
            ("normal", "normal", "linear"),
            ("displacement", "displacement", "linear"),
            ("roughness", "roughness", "linear"),
            ("specular", "specular_color", "linear"),
        ]
        try reviewRequire(materialChannels.count == expected.count, field: "material_channels", reason: "five canonical channels are required")
        for (index, material) in materialChannels.enumerated() {
            let field = "material_channels.\(index)"
            let tuple = expected[index]
            try reviewRequire(material.schemaVersion == ReviewContract.materialSchema, field: "\(field).schema_version", reason: "unsupported schema")
            try reviewRequire(material.channel == tuple.0 && material.manifestKey == tuple.1 && material.colorSpace == tuple.2, field: field, reason: "material identity differs")
            if let digest = material.sha256 {
                try reviewRequire(ReviewValidation.isSHA256(digest), field: "\(field).sha256", reason: "must be lowercase SHA-256")
            }
            let expectedStatus: String
            if let logicalName = material.artifactLogicalName {
                expectedStatus = "sealed_artifact"
                try reviewRequire(ReviewValidation.isIdentifier(logicalName), field: "\(field).artifact_logical_name", reason: "identifier is invalid")
                try reviewRequire(artifactsByName[logicalName]?.sha256 == material.sha256, field: field, reason: "material artifact hash is not bound")
            } else if material.sha256 != nil {
                expectedStatus = "hash_reference_only"
            } else {
                expectedStatus = "unavailable"
            }
            try reviewRequire(material.status == expectedStatus, field: "\(field).status", reason: "status is inconsistent")
            try reviewRequire(material.isolatable == (material.artifactLogicalName != nil), field: "\(field).isolatable", reason: "isolation claim is inconsistent")
            try reviewRequire(!material.measured && !material.productionValidated && material.approvalStatus == "unapproved", field: field, reason: "material claim must remain fail closed")
        }
    }

    private func validateCorrection() throws {
        let correction = correctionEligibility
        try reviewRequire(correction.schemaVersion == ReviewContract.correctionSchema, field: "correction_eligibility.schema_version", reason: "unsupported schema")
        try reviewRequire(!correction.candidateRequestEligible, field: "correction_eligibility.candidate_request_eligible", reason: "correction is not available in U1a")
        try reviewRequire(correction.candidateLayerID == .authoredCorrection, field: "correction_eligibility.candidate_layer_id", reason: "must identify authored_correction")
        try reviewRequire(correction.requiredParentRevisionID == layers.last?.revisionID, field: "correction_eligibility.required_parent_revision_id", reason: "must bind the final revision")
        try reviewRequire(correction.selectionRequiresExactSourcePTS && correction.immutableRevisionRequired, field: "correction_eligibility", reason: "exact immutable source selection is required")
        try reviewRequire(correction.undoRedoMode == "append_only_revision_selection", field: "correction_eligibility.undo_redo_mode", reason: "unsupported mode")
        try reviewRequire(correction.protectedAnchorClasses == ReviewContract.protectedAnchorClasses, field: "correction_eligibility.protected_anchor_classes", reason: "protected anchors differ")
        try reviewRequire(!correction.writerImplemented && !correction.productionRevisionEligible && !correction.humanReviewRecorded, field: "correction_eligibility", reason: "correction capability must remain disabled")
        try reviewRequire(correction.approvalStatus == "unapproved" && !correction.productionValidated, field: "correction_eligibility", reason: "correction must remain unapproved")
        try reviewRequire(correction.reasonCodes == ReviewContract.correctionReasonCodes, field: "correction_eligibility.reason_codes", reason: "fail-closed reasons differ")
    }

    private func validateFailClosedClaims() throws {
        let expectedClaims = ReviewClaims(
            artifactLedgerBytesVerified: true,
            exactRationalPTSClockVerified: true,
            manifestSignatureVerified: false,
            motionConsumptionIndependentlyVerified: false,
            materialsApproved: false,
            correctionApproved: false,
            performanceApproved: false,
            productionValidated: false,
            publishable: false
        )
        try reviewRequire(claims == expectedClaims, field: "claims", reason: "claims must remain fail closed")
        try reviewRequire(limitations == ReviewContract.limitations, field: "limitations", reason: "limitations differ from v1")
        let expectedBridge = ReviewBridgeContract(
            schemaVersion: ReviewContract.bridgeSchema,
            allowedMessageTypes: ["cursor", "layer", "selection", "revision"],
            messageVersionRequired: true,
            arbitraryScriptMessagesAllowed: false,
            productionCommandsEnabled: false
        )
        try reviewRequire(bridge == expectedBridge, field: "bridge", reason: "bridge capability contract differs")
    }
}

extension ReviewBundle.CodingKeys: CaseIterable {}
