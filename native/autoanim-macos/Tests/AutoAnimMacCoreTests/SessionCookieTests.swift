import Foundation
import Testing
@testable import AutoAnimMacCore

@Suite("Web viewer session cookie")
struct SessionCookieTests {
    @Test("Cookie is loopback-scoped, inaccessible to scripts, strict, and ephemeral")
    func secureCookiePolicy() throws {
        let token = String(repeating: "a", count: 64)
        let cookie = try #require(SessionCookie.make(token: token))

        #expect(cookie.name == "autoanim_session")
        #expect(cookie.value == token)
        #expect(cookie.domain == "127.0.0.1")
        #expect(cookie.path == "/")
        #expect(cookie.isHTTPOnly)
        #expect(cookie.sameSitePolicy == .sameSiteStrict)
        #expect(cookie.isSessionOnly)
        #expect(cookie.expiresDate == nil)
    }
}
