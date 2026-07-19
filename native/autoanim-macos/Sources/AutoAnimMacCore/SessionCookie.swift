import Foundation

public enum SessionCookie {
    public static let name = "autoanim_session"

    public static func make(token: String) -> HTTPCookie? {
        HTTPCookie(properties: [
            .domain: "127.0.0.1",
            .path: "/",
            .name: name,
            .value: token,
            .discard: "TRUE",
            .sameSitePolicy: HTTPCookieStringPolicy.sameSiteStrict,
            HTTPCookiePropertyKey("HttpOnly"): "TRUE",
        ])
    }
}
