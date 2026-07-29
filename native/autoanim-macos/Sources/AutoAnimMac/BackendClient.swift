import AutoAnimMacCore
import Foundation

struct BackendClient: Sendable {
    let endpoint: LoopbackEndpoint
    let token: String

    private static let maximumReviewBundleBytes = 8 * 1_024 * 1_024

    private func authenticatedData(for request: URLRequest) async throws -> (Data, URLResponse) {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.urlCache = nil
        let session = URLSession(
            configuration: configuration,
            delegate: RejectRedirectsDelegate(),
            delegateQueue: nil
        )
        defer { session.finishTasksAndInvalidate() }
        return try await session.data(for: request)
    }

    private func request(path: String, queryItems: [URLQueryItem] = []) throws -> URLRequest {
        let base = try endpoint.url(path: path)
        guard var components = URLComponents(url: base, resolvingAgainstBaseURL: false) else {
            throw BackendClientError.invalidResponse
        }
        components.queryItems = queryItems.isEmpty ? nil : queryItems
        guard let url = components.url else { throw BackendClientError.invalidResponse }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        request.timeoutInterval = 20
        request.setValue(token, forHTTPHeaderField: "X-AutoAnim-Token")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        return request
    }

    private func data(path: String, queryItems: [URLQueryItem] = []) async throws -> Data {
        let request = try request(path: path, queryItems: queryItems)
        let (data, response) = try await authenticatedData(for: request)
        guard let http = response as? HTTPURLResponse,
              http.statusCode == 200,
              responseMatchesRequest(http, request: request)
        else {
            throw BackendClientError.invalidResponse
        }
        return data
    }

    func health() async throws -> HealthReport {
        try ServiceDecoding.health(from: await data(path: "/api/health"))
    }

    func recentJobs() async throws -> [JobSummary] {
        try ServiceDecoding.jobs(
            from: await data(
                path: "/api/jobs",
                queryItems: [URLQueryItem(name: "limit", value: "50")]
            )
        ).jobs
    }

    func projects() async throws -> [ProductionProjectRecord] {
        try ServiceDecoding.projects(
            from: await data(
                path: "/api/projects",
                queryItems: [URLQueryItem(name: "limit", value: "200")]
            )
        ).projects
    }

    func shots(projectID: String) async throws -> [ProductionShotRecord] {
        guard JobSummary.validJobID(projectID) else {
            throw BackendClientError.invalidResponse
        }
        return try ServiceDecoding.shots(
            from: await data(
                path: "/api/projects/\(projectID)/shots",
                queryItems: [URLQueryItem(name: "limit", value: "200")]
            )
        ).shots
    }

    func characters() async throws -> [ProductionCharacterRecord] {
        try ServiceDecoding.characters(
            from: await data(
                path: "/api/characters",
                queryItems: [URLQueryItem(name: "limit", value: "200")]
            )
        ).characters
    }

    func productionLibrary() async throws -> ProductionLibrarySnapshot {
        async let fetchedProjects = projects()
        async let fetchedCharacters = characters()
        let (projects, characters) = try await (fetchedProjects, fetchedCharacters)
        let shots = try await withThrowingTaskGroup(
            of: (String, [ProductionShotRecord]).self
        ) { group in
            for project in projects {
                group.addTask {
                    (
                        project.projectID,
                        try await self.shots(projectID: project.projectID)
                    )
                }
            }
            var byProject: [String: [ProductionShotRecord]] = [:]
            for try await (projectID, values) in group {
                byProject[projectID] = values
            }
            return projects.flatMap { byProject[$0.projectID] ?? [] }
        }
        return ProductionLibrarySnapshot(
            projects: projects,
            characters: characters,
            shots: shots
        )
    }

    func createProject(name: String, description: String?) async throws -> ProductionProjectRecord {
        let payload = try await post(
            path: "/api/projects",
            body: CreateProjectRequest(name: name, description: description)
        )
        return try JSONDecoder().decode(ProductionProjectRecord.self, from: payload)
    }

    func createShot(
        projectID: String,
        name: String,
        characterID: String,
        characterRevisionID: String,
        description: String?
    ) async throws -> ProductionShotRecord {
        guard JobSummary.validJobID(projectID) else {
            throw BackendClientError.invalidResponse
        }
        let payload = try await post(
            path: "/api/projects/\(projectID)/shots",
            body: CreateShotRequest(
                name: name,
                characterID: characterID,
                characterRevisionID: characterRevisionID,
                description: description
            )
        )
        return try JSONDecoder().decode(ProductionShotRecord.self, from: payload)
    }

    func jobProvenance(jobID: String) async throws -> JobProvenanceRecord {
        guard JobSummary.validJobID(jobID) else { throw BackendClientError.invalidResponse }
        return try ServiceDecoding.jobProvenance(
            from: await data(path: "/api/jobs/\(jobID)")
        )
    }

    func createTake(
        projectID: String,
        shotID: String,
        name: String,
        mediaKind: String,
        input: JobInputProvenance
    ) async throws -> ProductionTakeRecord {
        guard JobSummary.validJobID(projectID), JobSummary.validJobID(shotID) else {
            throw BackendClientError.invalidResponse
        }
        let payload = try await post(
            path: "/api/projects/\(projectID)/shots/\(shotID)/takes",
            body: CreateTakeRequest(
                name: name,
                mediaKind: mediaKind,
                sourceName: input.name,
                sourceSHA256: input.sha256,
                sourceBytes: input.bytes,
                sourceMediaType: input.mediaType
            )
        )
        return try JSONDecoder().decode(ProductionTakeRecord.self, from: payload)
    }

    func linkJob(
        projectID: String,
        shotID: String,
        takeID: String,
        jobID: String
    ) async throws -> ProductionShotVersionRecord {
        guard JobSummary.validJobID(projectID), JobSummary.validJobID(shotID),
              JobSummary.validJobID(takeID), JobSummary.validJobID(jobID)
        else { throw BackendClientError.invalidResponse }
        let payload = try await post(
            path: "/api/projects/\(projectID)/shots/\(shotID)/takes/\(takeID)/jobs",
            body: LinkJobRequest(jobID: jobID)
        )
        return try JSONDecoder().decode(ProductionShotVersionRecord.self, from: payload)
    }

    func reviewBundle(jobID: String) async throws -> ReviewBundle {
        guard JobSummary.validJobID(jobID) else {
            throw BackendClientError.invalidResponse
        }
        let request = try request(path: "/api/jobs/\(jobID)/review-bundle")
        let (payload, response) = try await authenticatedData(for: request)
        guard
            let http = response as? HTTPURLResponse,
            http.statusCode == 200,
            responseMatchesRequest(http, request: request),
            http.mimeType == "application/json",
            http.expectedContentLength <= Int64(Self.maximumReviewBundleBytes),
            !payload.isEmpty,
            payload.count <= Self.maximumReviewBundleBytes
        else {
            throw BackendClientError.invalidResponse
        }
        let bundle = try ReviewBundle.decodeStrict(from: payload)
        guard
            bundle.sourceManifest.jobID == jobID,
            http.value(
                forHTTPHeaderField: "X-AutoAnim-Review-Bundle-SHA256"
            ) == bundle.bundleSHA256
        else {
            throw BackendClientError.invalidResponse
        }
        return bundle
    }

    private func responseMatchesRequest(
        _ response: HTTPURLResponse,
        request: URLRequest
    ) -> Bool {
        guard let requestedURL = request.url, let responseURL = response.url else {
            return false
        }
        return responseURL.absoluteString == requestedURL.absoluteString
    }

    private func post<Body: Encodable>(path: String, body: Body) async throws -> Data {
        var request = try request(path: path)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)
        let (payload, response) = try await authenticatedData(for: request)
        guard let http = response as? HTTPURLResponse,
              http.statusCode == 201,
              responseMatchesRequest(http, request: request),
              !payload.isEmpty
        else {
            throw BackendClientError.invalidResponse
        }
        return payload
    }
}

private struct CreateProjectRequest: Encodable {
    let name: String
    let description: String?
}

private struct CreateShotRequest: Encodable {
    let name: String
    let characterID: String
    let characterRevisionID: String
    let description: String?

    enum CodingKeys: String, CodingKey {
        case name
        case characterID = "character_id"
        case characterRevisionID = "character_revision_id"
        case description
    }
}

private struct CreateTakeRequest: Encodable {
    let name: String
    let mediaKind: String
    let sourceName: String
    let sourceSHA256: String
    let sourceBytes: Int
    let sourceMediaType: String?

    enum CodingKeys: String, CodingKey {
        case name
        case mediaKind = "media_kind"
        case sourceName = "source_name"
        case sourceSHA256 = "source_sha256"
        case sourceBytes = "source_bytes"
        case sourceMediaType = "source_media_type"
    }
}

private struct LinkJobRequest: Encodable {
    let jobID: String
    enum CodingKeys: String, CodingKey { case jobID = "job_id" }
}

private final class RejectRedirectsDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}

enum BackendClientError: LocalizedError {
    case invalidResponse

    var errorDescription: String? {
        "The authenticated source runtime returned an invalid response."
    }
}
