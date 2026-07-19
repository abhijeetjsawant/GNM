import AutoAnimMacCore
import SwiftUI
import WebKit

struct AuthenticatedViewer: NSViewRepresentable {
    let endpoint: LoopbackEndpoint
    let token: String
    let jobID: String

    func makeCoordinator() -> Coordinator {
        Coordinator(endpoint: endpoint)
    }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.setValue(false, forKey: "drawsBackground")
        load(webView, coordinator: context.coordinator)
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        guard context.coordinator.loadedJobID != jobID else { return }
        context.coordinator.endpoint = endpoint
        load(webView, coordinator: context.coordinator)
    }

    private func load(_ webView: WKWebView, coordinator: Coordinator) {
        guard let cookie = SessionCookie.make(token: token) else { return }
        guard let url = try? endpoint.viewerURL(jobID: jobID) else { return }
        coordinator.loadedJobID = jobID
        webView.configuration.websiteDataStore.httpCookieStore.setCookie(cookie) {
            var request = URLRequest(url: url)
            request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
            request.setValue(token, forHTTPHeaderField: "X-AutoAnim-Token")
            webView.load(request)
        }
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        var endpoint: LoopbackEndpoint
        var loadedJobID: String?

        init(endpoint: LoopbackEndpoint) {
            self.endpoint = endpoint
        }

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
        ) {
            guard let url = navigationAction.request.url else {
                decisionHandler(.cancel)
                return
            }
            let allowed = url.scheme == endpoint.baseURL.scheme
                && url.host == endpoint.baseURL.host
                && url.port == endpoint.baseURL.port
            decisionHandler(allowed ? .allow : .cancel)
        }
    }
}
