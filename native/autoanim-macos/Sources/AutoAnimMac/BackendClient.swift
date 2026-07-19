import AutoAnimMacCore
import Foundation

struct BackendClient: Sendable {
    let endpoint: LoopbackEndpoint
    let token: String

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
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieStorage = nil
        configuration.urlCache = nil
        let session = URLSession(configuration: configuration)
        let (data, response) = try await session.data(
            for: request(path: path, queryItems: queryItems)
        )
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
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
}

enum BackendClientError: LocalizedError {
    case invalidResponse

    var errorDescription: String? {
        "The authenticated source runtime returned an invalid response."
    }
}
