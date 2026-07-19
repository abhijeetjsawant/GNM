import Foundation
import Testing
@testable import AutoAnimMacCore

@Suite("Authenticated loopback endpoint")
struct LoopbackEndpointTests {
    @Test("Accepts and normalizes an IPv4 loopback URL")
    func acceptsLoopback() throws {
        let endpoint = try LoopbackEndpoint(
            validating: #require(URL(string: "http://127.0.0.1:49152"))
        )
        #expect(endpoint.baseURL.absoluteString == "http://127.0.0.1:49152/")
        #expect(try endpoint.url(path: "/api/health").absoluteString == "http://127.0.0.1:49152/api/health")
    }

    @Test(
        "Rejects non-loopback, credentialed, fixed-path, or non-HTTP URLs",
        arguments: [
            "https://127.0.0.1:49152/",
            "http://localhost:49152/",
            "http://0.0.0.0:49152/",
            "http://user@127.0.0.1:49152/",
            "http://127.0.0.1:49152/base",
            "http://127.0.0.1/",
            "http://127.0.0.1:49152/?token=leak",
        ]
    )
    func rejectsInvalidBase(value: String) {
        #expect(throws: LoopbackEndpointError.self) {
            _ = try LoopbackEndpoint(validating: #require(URL(string: value)))
        }
    }

    @Test("Rejects query and authority injection in route paths")
    func rejectsInvalidPath() throws {
        let endpoint = try LoopbackEndpoint(
            validating: #require(URL(string: "http://127.0.0.1:49152/"))
        )
        for path in ["api/health", "//attacker.test/api", "/api/jobs?token=leak", "/api/jobs#fragment"] {
            #expect(throws: LoopbackEndpointError.self) {
                _ = try endpoint.url(path: path)
            }
        }
    }

    @Test("Allows only canonical AutoAnim job identifiers in viewer URLs")
    func validatesViewerJobID() throws {
        let endpoint = try LoopbackEndpoint(
            validating: #require(URL(string: "http://127.0.0.1:49152/"))
        )
        let valid = "01kxwwdq8gqrsrzycjc3c3kjy9"
        #expect(try endpoint.viewerURL(jobID: valid).path == "/api/jobs/\(valid)/viewer")
        #expect(throws: LoopbackEndpointError.self) {
            _ = try endpoint.viewerURL(jobID: "../../etc/passwd")
        }
    }
}
