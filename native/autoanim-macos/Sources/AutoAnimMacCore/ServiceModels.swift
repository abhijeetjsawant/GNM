import Foundation

public struct LoopbackEndpoint: Equatable, Sendable {
    public let baseURL: URL

    public init(validating url: URL) throws {
        guard
            url.scheme == "http",
            url.host == "127.0.0.1",
            let port = url.port,
            (1...65_535).contains(port),
            url.user == nil,
            url.password == nil,
            url.query == nil,
            url.fragment == nil,
            url.path.isEmpty || url.path == "/"
        else {
            throw LoopbackEndpointError.invalidURL
        }
        var components = URLComponents()
        components.scheme = "http"
        components.host = "127.0.0.1"
        components.port = port
        components.path = "/"
        guard let normalized = components.url else {
            throw LoopbackEndpointError.invalidURL
        }
        self.baseURL = normalized
    }

    public func url(path: String) throws -> URL {
        guard
            path.hasPrefix("/"),
            !path.hasPrefix("//"),
            !path.contains("?"),
            !path.contains("#")
        else {
            throw LoopbackEndpointError.invalidPath
        }
        return baseURL.appending(path: String(path.dropFirst()))
    }

    public func viewerURL(jobID: String) throws -> URL {
        guard JobSummary.validJobID(jobID) else {
            throw LoopbackEndpointError.invalidPath
        }
        return try url(path: "/api/jobs/\(jobID)/viewer")
    }

    public func reviewBundleURL(jobID: String) throws -> URL {
        guard JobSummary.validJobID(jobID) else {
            throw LoopbackEndpointError.invalidPath
        }
        return try url(path: "/api/jobs/\(jobID)/review-bundle")
    }
}

public enum LoopbackEndpointError: Error, Equatable {
    case invalidURL
    case invalidPath
}

public struct BackendReadyEvent: Codable, Equatable, Sendable {
    public let event: String
    public let url: URL
    public let sourceRuntimeDependent: Bool

    enum CodingKeys: String, CodingKey {
        case event
        case url
        case sourceRuntimeDependent = "source_runtime_dependent"
    }
}

public struct HealthReport: Codable, Equatable, Sendable {
    public let status: String
    public let checks: [String: HealthCheck]
    public let versions: [String: String]
}

public struct HealthCheck: Codable, Equatable, Sendable {
    public let ready: Bool
    public let detail: String
    public let version: String?

    public init(ready: Bool, detail: String, version: String? = nil) {
        self.ready = ready
        self.detail = detail
        self.version = version
    }
}

public struct RecentJobsResponse: Codable, Equatable, Sendable {
    public let jobs: [JobSummary]
}

public struct JobSummary: Codable, Equatable, Identifiable, Sendable {
    public let jobID: String
    public let kind: String
    public let status: String
    public let createdAt: String?
    public let updatedAt: String?
    public let input: JobInputSummary
    public let warningCount: Int
    public let viewable: Bool

    public var id: String { jobID }

    enum CodingKeys: String, CodingKey {
        case jobID = "job_id"
        case kind
        case status
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case input
        case warningCount = "warning_count"
        case viewable
    }

    public static func validJobID(_ value: String) -> Bool {
        let alphabet = Set("0123456789abcdefghjkmnpqrstvwxyz")
        return value.count == 26 && value.allSatisfy { alphabet.contains($0) }
    }
}

public struct JobInputSummary: Codable, Equatable, Sendable {
    public let name: String
    public let mediaType: String?

    enum CodingKeys: String, CodingKey {
        case name
        case mediaType = "media_type"
    }
}

public struct JobInputProvenance: Codable, Equatable, Sendable {
    public let name: String
    public let sha256: String
    public let bytes: Int
    public let mediaType: String?

    enum CodingKeys: String, CodingKey {
        case name
        case sha256
        case bytes
        case mediaType = "media_type"
    }
}

public struct JobProvenanceRecord: Codable, Equatable, Sendable {
    public let jobID: String
    public let kind: String
    public let status: String
    public let input: JobInputProvenance
    public let configuration: JobCharacterConfiguration

    enum CodingKeys: String, CodingKey {
        case jobID = "job_id"
        case kind
        case status
        case input
        case configuration
    }
}

public struct JobCharacterConfiguration: Codable, Equatable, Sendable {
    public let characterID: String?
    public let characterRevisionID: String?

    enum CodingKeys: String, CodingKey {
        case characterID = "character_id"
        case characterRevisionID = "character_revision_id"
    }
}

public enum ServiceDecoding {
    public static func readyEvent(from line: String) throws -> BackendReadyEvent {
        try JSONDecoder().decode(BackendReadyEvent.self, from: Data(line.utf8))
    }

    public static func health(from data: Data) throws -> HealthReport {
        try JSONDecoder().decode(HealthReport.self, from: data)
    }

    public static func jobs(from data: Data) throws -> RecentJobsResponse {
        try JSONDecoder().decode(RecentJobsResponse.self, from: data)
    }

    public static func jobProvenance(from data: Data) throws -> JobProvenanceRecord {
        try JSONDecoder().decode(JobProvenanceRecord.self, from: data)
    }

    public static func projects(from data: Data) throws -> ProductionProjectsResponse {
        try JSONDecoder().decode(ProductionProjectsResponse.self, from: data)
    }

    public static func shots(from data: Data) throws -> ProductionShotsResponse {
        try JSONDecoder().decode(ProductionShotsResponse.self, from: data)
    }

    public static func characters(from data: Data) throws -> ProductionCharactersResponse {
        try JSONDecoder().decode(ProductionCharactersResponse.self, from: data)
    }
}
