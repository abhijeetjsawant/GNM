import Foundation

public enum ProductionItemStatus: String, Codable, CaseIterable, Equatable, Sendable {
    case active
    case ready
    case review
    case blocked
    case archived

    public var title: String {
        switch self {
        case .active: "In production"
        case .ready: "Ready"
        case .review: "Needs review"
        case .blocked: "Blocked"
        case .archived: "Archived"
        }
    }
}

public struct ProductionProjectRecord: Codable, Equatable, Identifiable, Sendable {
    public let schemaVersion: String
    public let recordType: String
    public let projectID: String
    public let name: String
    public let description: String?
    public let lifecycle: String
    public let createdAt: String
    public let updatedAt: String
    public let shotCount: Int
    public let shotIDs: [String]

    public var id: String { projectID }

    public var status: ProductionItemStatus {
        lifecycle == "active" ? .active : .archived
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case recordType = "record_type"
        case projectID = "project_id"
        case name
        case description
        case lifecycle
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case shotCount = "shot_count"
        case shotIDs = "shot_ids"
    }
}

public struct ProductionCharacterRevisionPin: Codable, Equatable, Sendable {
    public let characterID: String
    public let revisionID: String
    public let revisionManifestSHA256: String
    public let identitySHA256: String
    public let textureSHA256: String?
    public let textureUVsArraySHA256: String?
    public let materialManifestSHA256: String
    public let usageScope: String

    enum CodingKeys: String, CodingKey {
        case characterID = "character_id"
        case revisionID = "revision_id"
        case revisionManifestSHA256 = "revision_manifest_sha256"
        case identitySHA256 = "identity_sha256"
        case textureSHA256 = "texture_sha256"
        case textureUVsArraySHA256 = "texture_uvs_array_sha256"
        case materialManifestSHA256 = "material_manifest_sha256"
        case usageScope = "usage_scope"
    }
}

public struct ProductionShotRecord: Codable, Equatable, Identifiable, Sendable {
    public let schemaVersion: String
    public let recordType: String
    public let projectID: String
    public let shotID: String
    public let name: String
    public let description: String?
    public let lifecycle: String
    public let characterRevision: ProductionCharacterRevisionPin
    public let createdAt: String
    public let updatedAt: String
    public let takeCount: Int
    public let takeIDs: [String]
    public let versionCount: Int
    public let versionIDs: [String]
    public let latestVersionID: String?

    public var id: String { shotID }

    public var status: ProductionItemStatus {
        switch lifecycle {
        case "performance": .active
        case "ready": .ready
        case "setup": .review
        default: .blocked
        }
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case recordType = "record_type"
        case projectID = "project_id"
        case shotID = "shot_id"
        case name
        case description
        case lifecycle
        case characterRevision = "character_revision"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case takeCount = "take_count"
        case takeIDs = "take_ids"
        case versionCount = "version_count"
        case versionIDs = "version_ids"
        case latestVersionID = "latest_version_id"
    }
}

public struct ProductionTakeRecord: Codable, Equatable, Identifiable, Sendable {
    public let takeID: String
    public let lifecycle: String

    public var id: String { takeID }

    enum CodingKeys: String, CodingKey {
        case takeID = "take_id"
        case lifecycle
    }
}

public struct ProductionShotVersionRecord: Codable, Equatable, Identifiable, Sendable {
    public let shotVersionID: String
    public let jobID: String
    public let ordinal: Int

    public var id: String { shotVersionID }

    enum CodingKeys: String, CodingKey {
        case shotVersionID = "shot_version_id"
        case jobID = "job_id"
        case ordinal
    }
}

public struct ProductionCharacterRecord: Codable, Equatable, Identifiable, Sendable {
    public let characterID: String
    public let name: String
    public let createdAt: String
    public let updatedAt: String
    public let currentRevisionID: String
    public let consentStatus: String
    public let consentScope: String
    public let appearanceStatus: String
    public let bodyStatus: String
    public let productionValidated: Bool
    public let materialRightsStatus: String
    public let currentMaterialRightsExpiresAt: String?

    public var id: String { characterID }

    public var status: ProductionItemStatus {
        if consentStatus != "active" || materialRightsStatus == "expired" {
            return .blocked
        }
        return productionValidated ? .ready : .review
    }

    enum CodingKeys: String, CodingKey {
        case characterID = "character_id"
        case name
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case currentRevisionID = "current_revision_id"
        case consentStatus = "consent_status"
        case consentScope = "consent_scope"
        case appearanceStatus = "appearance_status"
        case bodyStatus = "body_status"
        case productionValidated = "production_validated"
        case materialRightsStatus = "material_rights_status"
        case currentMaterialRightsExpiresAt = "current_material_rights_expires_at"
    }
}

public struct ProductionProjectsResponse: Codable, Equatable, Sendable {
    public let projects: [ProductionProjectRecord]
}

public struct ProductionShotsResponse: Codable, Equatable, Sendable {
    public let shots: [ProductionShotRecord]
}

public struct ProductionCharactersResponse: Codable, Equatable, Sendable {
    public let characters: [ProductionCharacterRecord]
}

public struct ProductionLibrarySnapshot: Equatable, Sendable {
    public let projects: [ProductionProjectRecord]
    public let characters: [ProductionCharacterRecord]
    public let shots: [ProductionShotRecord]

    public init(
        projects: [ProductionProjectRecord],
        characters: [ProductionCharacterRecord],
        shots: [ProductionShotRecord]
    ) {
        self.projects = projects
        self.characters = characters
        self.shots = shots
    }

    public func project(id: String?) -> ProductionProjectRecord? {
        guard let id else { return nil }
        return projects.first { $0.id == id }
    }

    public func character(id: String?) -> ProductionCharacterRecord? {
        guard let id else { return nil }
        return characters.first { $0.id == id }
    }

    public func shot(id: String?) -> ProductionShotRecord? {
        guard let id else { return nil }
        return shots.first { $0.id == id }
    }

    public func shots(projectID: String) -> [ProductionShotRecord] {
        shots.filter { $0.projectID == projectID }
    }

    public func shots(characterID: String) -> [ProductionShotRecord] {
        shots.filter { $0.characterRevision.characterID == characterID }
    }

    public static let empty = ProductionLibrarySnapshot(
        projects: [],
        characters: [],
        shots: []
    )
}
