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
