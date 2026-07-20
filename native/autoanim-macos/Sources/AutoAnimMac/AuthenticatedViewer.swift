import AutoAnimMacCore
import SwiftUI
import WebKit

struct AuthenticatedViewer: NSViewRepresentable {
    enum ReviewBridgeCommandResolution: Equatable {
        case retry(sequence: UInt64)
        case drop(sequence: UInt64)

        var sequence: UInt64 {
            switch self {
            case .retry(let sequence), .drop(let sequence):
                return sequence
            }
        }
    }

    struct ReviewBridgeBinding {
        let bundle: ReviewBundle
        let commands: [ReviewBridgeEnvelope]
        let commandResolution: ReviewBridgeCommandResolution?
        let onDocumentReset: () -> Void
        let onCommandAccepted: (ReviewBridgeEnvelope) -> Void
        let onReceive: (ReviewBridgeEnvelope) -> Void
        let onError: (ReviewBridgeError) -> Void

        init(
            bundle: ReviewBundle,
            commands: [ReviewBridgeEnvelope] = [],
            commandResolution: ReviewBridgeCommandResolution? = nil,
            onDocumentReset: @escaping () -> Void = {},
            onCommandAccepted: @escaping (ReviewBridgeEnvelope) -> Void = { _ in },
            onReceive: @escaping (ReviewBridgeEnvelope) -> Void,
            onError: @escaping (ReviewBridgeError) -> Void = { _ in }
        ) {
            self.bundle = bundle
            self.commands = commands
            self.commandResolution = commandResolution
            self.onDocumentReset = onDocumentReset
            self.onCommandAccepted = onCommandAccepted
            self.onReceive = onReceive
            self.onError = onError
        }

        init(
            bundle: ReviewBundle,
            command: ReviewBridgeEnvelope?,
            commandResolution: ReviewBridgeCommandResolution? = nil,
            onDocumentReset: @escaping () -> Void = {},
            onCommandAccepted: @escaping (ReviewBridgeEnvelope) -> Void = { _ in },
            onReceive: @escaping (ReviewBridgeEnvelope) -> Void,
            onError: @escaping (ReviewBridgeError) -> Void = { _ in }
        ) {
            self.init(
                bundle: bundle,
                commands: command.map { [$0] } ?? [],
                commandResolution: commandResolution,
                onDocumentReset: onDocumentReset,
                onCommandAccepted: onCommandAccepted,
                onReceive: onReceive,
                onError: onError
            )
        }
    }

    enum ReviewBridgeError: Error, LocalizedError, Equatable {
        case invalidBinding(String)
        case rejectedMessage(String)
        case invalidCommandQueue(String)
        case rejectedCommand(jobID: String, sequence: UInt64, detail: String)
        case commandDeliveryFailed(jobID: String, sequence: UInt64, detail: String)
        case provisionalNavigationFailed(String)
        case navigationFailed(String)
        case webContentProcessTerminated

        var errorDescription: String? {
            switch self {
            case .invalidBinding(let detail):
                return "The review viewer binding is invalid: \(detail)"
            case .rejectedMessage(let detail):
                return "The review viewer message was rejected: \(detail)"
            case .invalidCommandQueue(let detail):
                return "The review command queue is invalid: \(detail)"
            case .rejectedCommand(let jobID, let sequence, let detail):
                return "Review command \(jobID)#\(sequence) was rejected: \(detail)"
            case .commandDeliveryFailed(let jobID, let sequence, let detail):
                return "Review command \(jobID)#\(sequence) was not accepted: \(detail)"
            case .provisionalNavigationFailed(let detail):
                return "The review viewer could not start loading: \(detail)"
            case .navigationFailed(let detail):
                return "The review viewer failed while loading: \(detail)"
            case .webContentProcessTerminated:
                return "The review viewer web-content process terminated."
            }
        }
    }

    let endpoint: LoopbackEndpoint
    let token: String
    let jobID: String
    let reviewBridge: ReviewBridgeBinding?

    init(
        endpoint: LoopbackEndpoint,
        token: String,
        jobID: String,
        reviewBridge: ReviewBridgeBinding? = nil
    ) {
        self.endpoint = endpoint
        self.token = token
        self.jobID = jobID
        self.reviewBridge = reviewBridge
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(endpoint: endpoint, jobID: jobID, reviewBridge: reviewBridge)
    }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.uiDelegate = context.coordinator
        webView.setValue(false, forKey: "drawsBackground")
        context.coordinator.attach(to: webView)
        load(webView, coordinator: context.coordinator)
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        let coordinator = context.coordinator
        coordinator.update(
            endpoint: endpoint,
            jobID: jobID,
            reviewBridge: reviewBridge,
            webView: webView
        )
        let reviewDocumentID = reviewBridge?.bundle.bundleSHA256
        if !coordinator.hasDocumentRequest(
            endpoint: endpoint,
            token: token,
            jobID: jobID,
            reviewDocumentID: reviewDocumentID
        ) {
            load(webView, coordinator: coordinator)
        } else {
            coordinator.deliverNextCommandIfPossible()
        }
    }

    static func dismantleNSView(_ webView: WKWebView, coordinator: Coordinator) {
        coordinator.detach(from: webView)
    }

    private func load(_ webView: WKWebView, coordinator: Coordinator) {
        guard let cookie = SessionCookie.make(token: token) else {
            coordinator.reportBindingError("the session token could not be represented as a cookie")
            return
        }
        guard let url = try? endpoint.viewerURL(jobID: jobID) else {
            coordinator.reportBindingError("the job does not produce a canonical viewer URL")
            return
        }
        let generation = coordinator.beginDocumentLoad(
            endpoint: endpoint,
            token: token,
            jobID: jobID,
            reviewDocumentID: reviewBridge?.bundle.bundleSHA256,
            viewerURL: url
        )
        webView.configuration.websiteDataStore.httpCookieStore.setCookie(cookie) {
            [weak webView, weak coordinator] in
            guard let coordinator else { return }
            coordinator.performAfterDocumentReset(generation: generation) {
                [weak webView, weak coordinator] in
                guard let webView, let coordinator,
                      coordinator.canIssueNavigation(generation: generation, in: webView)
                else { return }
                var request = URLRequest(url: url)
                request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
                request.setValue(token, forHTTPHeaderField: "X-AutoAnim-Token")
                guard let navigation = webView.load(request) else {
                    coordinator.failBeforeNavigation(
                        generation: generation,
                        detail: "WebKit did not create a navigation"
                    )
                    return
                }
                coordinator.didIssueNavigation(navigation, generation: generation)
            }
        }
    }

    final class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler {
        private static let handlerName = "autoanimReview"
        private static let maximumCommandSnapshotCount = 256
        private static let receiveCommandFunction =
            "return window.autoanimReview.receive(command);"

        private var endpoint: LoopbackEndpoint
        private var jobID: String
        private weak var webView: WKWebView?
        private var reviewBridge: ReviewBridgeBinding?
        private var bindingIsUsable = false
        private var validatedBinding: BindingIdentity?
        private var handlerRegistered = false

        private var documentGeneration: UInt64 = 0
        private var activeLoad: ActiveDocumentLoad?
        private var loadedDocument: DocumentIdentity?
        private var failedDocument: DocumentIdentity?
        private var expectedViewerURL: URL?
        private var resetDeliveredGeneration: UInt64?
        private var pendingNavigationAfterReset: (() -> Void)?
        private var acceptingMessages = false
        private var documentReady = false

        private var commandQueue: [ReviewBridgeEnvelope] = []
        private var seenCommands: Set<CommandIdentity> = []
        private var commandsSuppressedAtReset: Set<CommandIdentity> = []
        private var highestQueuedSequence: UInt64?
        private var inFlightCommand: CommandIdentity?
        private var failedCommand: CommandIdentity?
        private var lastObservedResolution: ReviewBridgeCommandResolution?

        init(
            endpoint: LoopbackEndpoint,
            jobID: String,
            reviewBridge: ReviewBridgeBinding?
        ) {
            self.endpoint = endpoint
            self.jobID = jobID
            self.reviewBridge = reviewBridge
        }

        func attach(to webView: WKWebView) {
            self.webView = webView
            validateReviewBinding()
            configureMessageHandler(in: webView)
        }

        func detach(from webView: WKWebView) {
            documentGeneration &+= 1
            if handlerRegistered {
                webView.configuration.userContentController.removeScriptMessageHandler(
                    forName: Self.handlerName
                )
            }
            handlerRegistered = false
            activeLoad = nil
            loadedDocument = nil
            failedDocument = nil
            resetDeliveredGeneration = nil
            pendingNavigationAfterReset = nil
            acceptingMessages = false
            documentReady = false
            resetCommandQueueForNewDocument()
            webView.stopLoading()
            webView.navigationDelegate = nil
            webView.uiDelegate = nil
            self.webView = nil
            reviewBridge = nil
        }

        func update(
            endpoint: LoopbackEndpoint,
            jobID: String,
            reviewBridge: ReviewBridgeBinding?,
            webView: WKWebView
        ) {
            self.endpoint = endpoint
            self.jobID = jobID
            self.reviewBridge = reviewBridge
            validateReviewBinding()
            configureMessageHandler(in: webView)
            observeResolution(reviewBridge?.commandResolution)
            observeCommands(reviewBridge?.commands ?? [])
        }

        func hasDocumentRequest(
            endpoint: LoopbackEndpoint,
            token: String,
            jobID: String,
            reviewDocumentID: String?
        ) -> Bool {
            let identity = DocumentIdentity(
                endpoint: endpoint,
                token: token,
                jobID: jobID,
                reviewDocumentID: reviewDocumentID
            )
            return activeLoad?.identity == identity
                || loadedDocument == identity
                || failedDocument == identity
        }

        func beginDocumentLoad(
            endpoint: LoopbackEndpoint,
            token: String,
            jobID: String,
            reviewDocumentID: String?,
            viewerURL: URL
        ) -> UInt64 {
            documentGeneration &+= 1
            let identity = DocumentIdentity(
                endpoint: endpoint,
                token: token,
                jobID: jobID,
                reviewDocumentID: reviewDocumentID
            )
            webView?.stopLoading()
            self.endpoint = endpoint
            self.jobID = jobID
            activeLoad = ActiveDocumentLoad(
                generation: documentGeneration,
                identity: identity,
                navigation: nil
            )
            loadedDocument = nil
            failedDocument = nil
            expectedViewerURL = viewerURL
            resetDeliveredGeneration = bindingIsUsable ? nil : documentGeneration
            pendingNavigationAfterReset = nil
            acceptingMessages = false
            documentReady = false
            resetCommandQueueForNewDocument()
            if bindingIsUsable {
                scheduleDocumentReset(generation: documentGeneration)
            }
            return documentGeneration
        }

        func performAfterDocumentReset(
            generation: UInt64,
            operation: @escaping () -> Void
        ) {
            guard activeLoad?.generation == generation else { return }
            if !bindingIsUsable || resetDeliveredGeneration == generation {
                operation()
            } else {
                pendingNavigationAfterReset = operation
            }
        }

        func canIssueNavigation(generation: UInt64, in webView: WKWebView) -> Bool {
            self.webView === webView
                && activeLoad?.generation == generation
                && activeLoad?.navigation == nil
                && (!bindingIsUsable || resetDeliveredGeneration == generation)
        }

        func didIssueNavigation(_ navigation: WKNavigation, generation: UInt64) {
            guard var activeLoad,
                  activeLoad.generation == generation,
                  activeLoad.navigation == nil
            else { return }
            activeLoad.navigation = navigation
            self.activeLoad = activeLoad
        }

        func failBeforeNavigation(generation: UInt64, detail: String) {
            guard activeLoad?.generation == generation else { return }
            markActiveDocumentFailed()
            reviewBridge?.onError(.provisionalNavigationFailed(detail))
        }

        func reportBindingError(_ detail: String) {
            reviewBridge?.onError(.invalidBinding(detail))
        }

        private func validateReviewBinding() {
            guard let reviewBridge else {
                bindingIsUsable = false
                validatedBinding = nil
                return
            }
            let identity = BindingIdentity(
                jobID: jobID,
                bundleSHA256: reviewBridge.bundle.bundleSHA256
            )
            guard identity != validatedBinding else { return }
            validatedBinding = identity
            do {
                try reviewBridge.bundle.validate()
                guard reviewBridge.bundle.sourceManifest.jobID == jobID else {
                    throw ReviewBridgeError.invalidBinding(
                        "the ReviewBundle belongs to a different job"
                    )
                }
                guard reviewBridge.bundle.revisionGraph.renderableRevisions.count == 1 else {
                    throw ReviewBridgeError.invalidBinding(
                        "the ReviewBundle must expose one renderable revision"
                    )
                }
                bindingIsUsable = true
            } catch let error as ReviewBridgeError {
                bindingIsUsable = false
                reviewBridge.onError(error)
            } catch {
                bindingIsUsable = false
                reviewBridge.onError(.invalidBinding(error.localizedDescription))
            }
        }

        private func configureMessageHandler(in webView: WKWebView) {
            if bindingIsUsable && !handlerRegistered {
                webView.configuration.userContentController.add(
                    self,
                    name: Self.handlerName
                )
                handlerRegistered = true
            } else if !bindingIsUsable && handlerRegistered {
                webView.configuration.userContentController.removeScriptMessageHandler(
                    forName: Self.handlerName
                )
                handlerRegistered = false
            }
        }

        private func scheduleDocumentReset(generation: UInt64) {
            guard bindingIsUsable,
                  let bindingIdentity = validatedBinding,
                  let callback = reviewBridge?.onDocumentReset
            else { return }
            DispatchQueue.main.async { [weak self] in
                guard let self,
                      self.documentGeneration == generation,
                      self.bindingIsUsable,
                      self.validatedBinding == bindingIdentity
                else { return }
                callback()
                guard self.documentGeneration == generation,
                      self.bindingIsUsable,
                      self.validatedBinding == bindingIdentity
                else { return }
                self.resetDeliveredGeneration = generation
                let navigation = self.pendingNavigationAfterReset
                self.pendingNavigationAfterReset = nil
                navigation?()
            }
        }

        private func boundRevision(in bundle: ReviewBundle) -> ReviewRenderableRevision? {
            guard bundle.revisionGraph.renderableRevisions.count == 1 else { return nil }
            return bundle.revisionGraph.renderableRevisions[0]
        }

        private func validateBoundEnvelope(
            _ envelope: ReviewBridgeEnvelope,
            direction: ReviewBridgeDirection
        ) throws {
            guard bindingIsUsable, let reviewBridge else {
                throw ReviewBridgeError.invalidBinding("review integration is disabled")
            }
            try envelope.validate(direction: direction)
            guard envelope.jobID == jobID,
                  envelope.jobID == reviewBridge.bundle.sourceManifest.jobID
            else {
                throw ReviewBridgeError.invalidBinding(
                    "the envelope belongs to a different job"
                )
            }
            guard let revision = boundRevision(in: reviewBridge.bundle) else {
                throw ReviewBridgeError.invalidBinding(
                    "the ReviewBundle has no unique renderable revision"
                )
            }
            let comparisonKey = reviewBridge.bundle.comparisonKey.comparisonKeySHA256
            switch envelope.payload {
            case .viewerReady(let payload):
                guard payload.comparisonKey == comparisonKey,
                      payload.revisionID == revision.revisionID
                else {
                    throw ReviewBridgeError.invalidBinding(
                        "viewer.ready does not match the bound comparison and revision"
                    )
                }
            case .revisionReady(let payload), .revisionSet(let payload):
                guard payload.comparisonKey == comparisonKey,
                      payload.revisionID == revision.revisionID
                else {
                    throw ReviewBridgeError.invalidBinding(
                        "the revision message does not match the bound comparison and revision"
                    )
                }
            default:
                break
            }
        }

        func userContentController(
            _ userContentController: WKUserContentController,
            didReceive message: WKScriptMessage
        ) {
            guard acceptingMessages,
                  resetDeliveredGeneration == documentGeneration,
                  message.name == Self.handlerName,
                  let webView,
                  message.webView === webView,
                  message.frameInfo.isMainFrame,
                  isExpectedOrigin(message.frameInfo.securityOrigin),
                  isExpectedViewerURL(message.frameInfo.request.url)
            else {
                reviewBridge?.onError(.rejectedMessage(
                    "the message did not originate from the active bound main-frame viewer"
                ))
                return
            }
            do {
                guard let object = message.body as? [String: Any],
                      JSONSerialization.isValidJSONObject(object)
                else {
                    throw ReviewBridgeError.rejectedMessage(
                        "the message body is not a JSON object"
                    )
                }
                let data = try JSONSerialization.data(
                    withJSONObject: object,
                    options: [.sortedKeys]
                )
                guard data.count <= ReviewBridgeEnvelope.maximumBytes else {
                    throw ReviewBridgeError.rejectedMessage(
                        "the message body exceeds 64 KiB"
                    )
                }
                let envelope = try ReviewBridgeEnvelope.decode(
                    data,
                    direction: .viewerToNative
                )
                try validateBoundEnvelope(envelope, direction: .viewerToNative)
                reviewBridge?.onReceive(envelope)
            } catch let error as ReviewBridgeError {
                reviewBridge?.onError(error)
            } catch {
                reviewBridge?.onError(.rejectedMessage(error.localizedDescription))
            }
        }

        private func observeCommands(_ commands: [ReviewBridgeEnvelope]) {
            guard commands.count <= Self.maximumCommandSnapshotCount else {
                reviewBridge?.onError(.invalidCommandQueue(
                    "a snapshot may contain at most \(Self.maximumCommandSnapshotCount) commands"
                ))
                return
            }
            let identities = commands.map(CommandIdentity.init)
            guard zip(identities, identities.dropFirst()).allSatisfy({ pair in
                pair.0.jobID == pair.1.jobID && pair.0.sequence < pair.1.sequence
            }) else {
                reviewBridge?.onError(.invalidCommandQueue(
                    "commands must have one job ID and strictly increasing sequences"
                ))
                return
            }
            let snapshotIdentities = Set(identities)
            commandsSuppressedAtReset.formIntersection(snapshotIdentities)
            guard commandsSuppressedAtReset.isEmpty else { return }
            for (command, identity) in zip(commands, identities) {
                guard !commandsSuppressedAtReset.contains(identity),
                      !seenCommands.contains(identity)
                else { continue }
                if let highestQueuedSequence,
                   command.sequence <= highestQueuedSequence
                {
                    seenCommands.insert(identity)
                    reviewBridge?.onError(.rejectedCommand(
                        jobID: command.jobID,
                        sequence: command.sequence,
                        detail: "sequence is not newer than the existing FIFO"
                    ))
                    continue
                }
                seenCommands.insert(identity)
                highestQueuedSequence = command.sequence
                commandQueue.append(command)
            }
            deliverNextCommandIfPossible()
        }

        private func observeResolution(_ resolution: ReviewBridgeCommandResolution?) {
            guard resolution != lastObservedResolution else { return }
            lastObservedResolution = resolution
            guard let resolution,
                  let failedCommand,
                  failedCommand.sequence == resolution.sequence,
                  commandQueue.first.map(CommandIdentity.init) == failedCommand
            else { return }
            self.failedCommand = nil
            switch resolution {
            case .retry:
                deliverNextCommandIfPossible()
            case .drop:
                commandQueue.removeFirst()
                deliverNextCommandIfPossible()
            }
        }

        func deliverNextCommandIfPossible() {
            guard documentReady,
                  bindingIsUsable,
                  inFlightCommand == nil,
                  failedCommand == nil,
                  let webView,
                  let command = commandQueue.first
            else { return }
            let identity = CommandIdentity(command)
            let generation = documentGeneration
            do {
                try validateBoundEnvelope(command, direction: .nativeToViewer)
                let data = try command.encoded()
                guard data.count <= ReviewBridgeEnvelope.maximumBytes else {
                    throw ReviewBridgeError.rejectedCommand(
                        jobID: command.jobID,
                        sequence: command.sequence,
                        detail: "the encoded command exceeds 64 KiB"
                    )
                }
                let object = try JSONSerialization.jsonObject(with: data)
                guard let dictionary = object as? [String: Any] else {
                    throw ReviewBridgeError.rejectedCommand(
                        jobID: command.jobID,
                        sequence: command.sequence,
                        detail: "the encoded command is not a JSON object"
                    )
                }
                inFlightCommand = identity
                webView.callAsyncJavaScript(
                    Self.receiveCommandFunction,
                    arguments: ["command": dictionary],
                    in: nil,
                    in: .page
                ) { [weak self] result in
                    guard let self,
                          self.documentGeneration == generation,
                          self.inFlightCommand == identity,
                          self.commandQueue.first.map(CommandIdentity.init) == identity
                    else { return }
                    self.inFlightCommand = nil
                    switch result {
                    case .success(let value) where value as? Bool == true:
                        self.commandQueue.removeFirst()
                        self.reviewBridge?.onCommandAccepted(command)
                        self.deliverNextCommandIfPossible()
                    case .success:
                        self.failCommand(
                            identity,
                            error: .commandDeliveryFailed(
                                jobID: identity.jobID,
                                sequence: identity.sequence,
                                detail: "the fixed viewer entrypoint rejected the command"
                            )
                        )
                    case .failure(let error):
                        self.failCommand(
                            identity,
                            error: .commandDeliveryFailed(
                                jobID: identity.jobID,
                                sequence: identity.sequence,
                                detail: error.localizedDescription
                            )
                        )
                    }
                }
            } catch let error as ReviewBridgeError {
                failCommand(identity, error: error)
            } catch {
                failCommand(
                    identity,
                    error: .rejectedCommand(
                        jobID: identity.jobID,
                        sequence: identity.sequence,
                        detail: error.localizedDescription
                    )
                )
            }
        }

        private func failCommand(_ identity: CommandIdentity, error: ReviewBridgeError) {
            guard commandQueue.first.map(CommandIdentity.init) == identity else { return }
            inFlightCommand = nil
            failedCommand = identity
            reviewBridge?.onError(error)
        }

        private func resetCommandQueueForNewDocument() {
            commandQueue.removeAll(keepingCapacity: true)
            seenCommands.removeAll(keepingCapacity: true)
            highestQueuedSequence = nil
            inFlightCommand = nil
            failedCommand = nil
            commandsSuppressedAtReset = Set(
                reviewBridge?.commands.map(CommandIdentity.init) ?? []
            )
            lastObservedResolution = reviewBridge?.commandResolution
        }

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
        ) {
            guard navigationAction.targetFrame?.isMainFrame == true,
                  let url = navigationAction.request.url,
                  isExpectedViewerURL(url)
            else {
                decisionHandler(.cancel)
                return
            }
            decisionHandler(.allow)
        }

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationResponse: WKNavigationResponse,
            decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void
        ) {
            guard navigationResponse.isForMainFrame,
                  let url = navigationResponse.response.url,
                  isExpectedViewerURL(url)
            else {
                decisionHandler(.cancel)
                return
            }
            decisionHandler(.allow)
        }

        func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!) {
            guard isActive(navigation), isExpectedViewerURL(webView.url) else { return }
            acceptingMessages = true
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            guard let activeLoad,
                  activeLoad.navigation === navigation,
                  isExpectedViewerURL(webView.url)
            else { return }
            loadedDocument = activeLoad.identity
            self.activeLoad = nil
            failedDocument = nil
            acceptingMessages = true
            documentReady = true
            deliverNextCommandIfPossible()
        }

        func webView(
            _ webView: WKWebView,
            didFailProvisionalNavigation navigation: WKNavigation!,
            withError error: Error
        ) {
            guard isActive(navigation) else { return }
            markActiveDocumentFailed()
            reviewBridge?.onError(.provisionalNavigationFailed(error.localizedDescription))
        }

        func webView(
            _ webView: WKWebView,
            didFail navigation: WKNavigation!,
            withError error: Error
        ) {
            guard isActive(navigation) else { return }
            markActiveDocumentFailed()
            reviewBridge?.onError(.navigationFailed(error.localizedDescription))
        }

        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            guard self.webView === webView else { return }
            documentGeneration &+= 1
            let document = loadedDocument ?? activeLoad?.identity
            activeLoad = nil
            loadedDocument = nil
            failedDocument = document
            resetDeliveredGeneration = nil
            pendingNavigationAfterReset = nil
            acceptingMessages = false
            documentReady = false
            resetCommandQueueForNewDocument()
            if bindingIsUsable {
                scheduleDocumentReset(generation: documentGeneration)
            }
            reviewBridge?.onError(.webContentProcessTerminated)
        }

        func webView(
            _ webView: WKWebView,
            createWebViewWith configuration: WKWebViewConfiguration,
            for navigationAction: WKNavigationAction,
            windowFeatures: WKWindowFeatures
        ) -> WKWebView? {
            nil
        }

        private func isActive(_ navigation: WKNavigation?) -> Bool {
            guard let navigation else { return false }
            return activeLoad?.navigation === navigation
        }

        private func markActiveDocumentFailed() {
            failedDocument = activeLoad?.identity
            activeLoad = nil
            loadedDocument = nil
            acceptingMessages = false
            documentReady = false
            inFlightCommand = nil
        }

        private func isExpectedOrigin(_ origin: WKSecurityOrigin) -> Bool {
            origin.protocol == endpoint.baseURL.scheme
                && origin.host == endpoint.baseURL.host
                && origin.port == endpoint.baseURL.port
        }

        private func isExpectedViewerURL(_ url: URL?) -> Bool {
            guard let url,
                  let expectedViewerURL,
                  let components = URLComponents(
                    url: url,
                    resolvingAgainstBaseURL: false
                  ),
                  let expectedComponents = URLComponents(
                    url: expectedViewerURL,
                    resolvingAgainstBaseURL: false
                  )
            else { return false }
            return url.scheme == expectedViewerURL.scheme
                && url.host == expectedViewerURL.host
                && url.port == expectedViewerURL.port
                && url.user == nil
                && url.password == nil
                && url.path == expectedViewerURL.path
                && components.percentEncodedPath == expectedComponents.percentEncodedPath
                && components.percentEncodedQuery == nil
                && components.fragment == nil
        }

        private struct BindingIdentity: Equatable {
            let jobID: String
            let bundleSHA256: String
        }

        private struct DocumentIdentity: Equatable {
            let endpoint: LoopbackEndpoint
            let token: String
            let jobID: String
            let reviewDocumentID: String?
        }

        private struct ActiveDocumentLoad {
            let generation: UInt64
            let identity: DocumentIdentity
            var navigation: WKNavigation?
        }

        private struct CommandIdentity: Equatable, Hashable {
            let jobID: String
            let sequence: UInt64

            init(_ envelope: ReviewBridgeEnvelope) {
                jobID = envelope.jobID
                sequence = envelope.sequence
            }
        }
    }
}
