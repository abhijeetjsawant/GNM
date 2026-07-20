import AutoAnimMacCore
import Foundation

@MainActor
final class AppModel: ObservableObject {
    @Published private(set) var configuration: RuntimeConfiguration?
    @Published private(set) var configurationError: String?
    @Published private(set) var health: HealthReport?
    @Published private(set) var jobs: [JobSummary] = []
    @Published private(set) var reviewBundles: [String: ReviewBundle] = [:]
    @Published private(set) var reviewBundleErrors: [String: String] = [:]
    @Published private(set) var loadingReviewJobIDs: Set<String> = []
    @Published var selectedJobID: String?
    @Published var selectedSection: LibrarySection = .jobs
    @Published private(set) var requestError: String?
    @Published private(set) var refreshing = false

    let supervisor: BackendSupervisor?

    private var runtimeEpoch: UInt64 = 0
    private var nextRequestID: UInt64 = 0
    private var activeRefreshRequestID: UInt64?
    private var activeDiscoveryRequestID: UInt64?
    private var refreshTask: Task<Void, Never>?
    private var reviewDiscoveryTask: Task<Void, Never>?
    private var reviewBundleTasks: [String: Task<ReviewBundle, Error>] = [:]
    private var reviewBundleRequestIDs: [String: UInt64] = [:]

    init(bundle: Bundle = .main) {
        do {
            let configuration = try RuntimeConfiguration.from(bundle: bundle)
            self.configuration = configuration
            let supervisor = BackendSupervisor(configuration: configuration)
            self.supervisor = supervisor
            supervisor.onReady = { [weak self] endpoint, token in
                self?.runtimeDidBecomeReady(endpoint: endpoint, token: token)
            }
            supervisor.onStopped = { [weak self] in
                self?.runtimeDidStop()
            }
        } catch {
            configuration = nil
            supervisor = nil
            configurationError = error.localizedDescription
        }
    }

    var selectedJob: JobSummary? {
        jobs.first { $0.jobID == selectedJobID }
    }

    var diagnostics: [RuntimeDiagnostic] {
        configuration?.diagnostics() ?? []
    }

    func start() {
        supervisor?.start()
    }

    func stop() {
        runtimeDidStop()
        supervisor?.stop()
    }

    func restart() {
        runtimeDidStop()
        supervisor?.restart()
    }

    func refresh() {
        guard let endpoint = supervisor?.endpoint, let token = supervisor?.token else { return }
        refreshTask?.cancel()
        let epoch = runtimeEpoch
        let requestID = allocateRequestID()
        activeRefreshRequestID = requestID
        refreshing = true
        requestError = nil
        let client = BackendClient(endpoint: endpoint, token: token)
        let task = Task { [weak self] in
            do {
                async let fetchedHealth = client.health()
                async let fetchedJobs = client.recentJobs()
                let (newHealth, newJobs) = try await (fetchedHealth, fetchedJobs)
                guard let self,
                      !Task.isCancelled,
                      self.isCurrentRuntime(epoch: epoch, endpoint: endpoint, token: token),
                      self.activeRefreshRequestID == requestID
                else { return }
                self.health = newHealth
                self.jobs = newJobs
                if self.selectedJobID == nil
                    || !newJobs.contains(where: { $0.jobID == self.selectedJobID }) {
                    self.selectedJobID = newJobs.first?.jobID
                }
            } catch {
                guard let self,
                      self.isCurrentRuntime(epoch: epoch, endpoint: endpoint, token: token),
                      self.activeRefreshRequestID == requestID
                else { return }
                if !Task.isCancelled && !Self.isCancellation(error) {
                    self.requestError = error.localizedDescription
                }
            }
            self?.finishRefresh(requestID: requestID)
        }
        refreshTask = task
    }

    func loadReviewBundle(jobID: String) async {
        guard reviewBundles[jobID] == nil,
              !loadingReviewJobIDs.contains(jobID),
              let endpoint = supervisor?.endpoint,
              let token = supervisor?.token
        else { return }
        let epoch = runtimeEpoch
        let requestID = allocateRequestID()
        let task = Task {
            try await BackendClient(
                endpoint: endpoint,
                token: token
            ).reviewBundle(jobID: jobID)
        }
        reviewBundleTasks[jobID] = task
        reviewBundleRequestIDs[jobID] = requestID
        loadingReviewJobIDs.insert(jobID)
        reviewBundleErrors.removeValue(forKey: jobID)
        defer { finishReviewBundleLoad(jobID: jobID, requestID: requestID) }
        do {
            let bundle = try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
            }
            guard !Task.isCancelled,
                  !task.isCancelled,
                  isCurrentRuntime(epoch: epoch, endpoint: endpoint, token: token),
                  reviewBundleRequestIDs[jobID] == requestID
            else { return }
            reviewBundles[jobID] = bundle
        } catch {
            guard !Task.isCancelled,
                  !task.isCancelled,
                  !Self.isCancellation(error),
                  isCurrentRuntime(epoch: epoch, endpoint: endpoint, token: token),
                  reviewBundleRequestIDs[jobID] == requestID
            else { return }
            reviewBundleErrors[jobID] = error.localizedDescription
        }
    }

    func discoverComparisonCandidates(for jobID: String) async {
        cancelReviewOperations()
        let epoch = runtimeEpoch
        let requestID = allocateRequestID()
        activeDiscoveryRequestID = requestID
        let task = Task { [weak self] in
            guard let self else { return }
            await self.performComparisonDiscovery(
                for: jobID,
                epoch: epoch,
                requestID: requestID
            )
        }
        reviewDiscoveryTask = task
        await withTaskCancellationHandler {
            await task.value
        } onCancel: {
            task.cancel()
        }
        if activeDiscoveryRequestID == requestID {
            activeDiscoveryRequestID = nil
            reviewDiscoveryTask = nil
        }
    }

    private func performComparisonDiscovery(
        for jobID: String,
        epoch: UInt64,
        requestID: UInt64
    ) async {
        guard isCurrentDiscovery(epoch: epoch, requestID: requestID) else { return }
        await loadReviewBundle(jobID: jobID)
        guard isCurrentDiscovery(epoch: epoch, requestID: requestID),
              reviewBundles[jobID] != nil
        else { return }
        let candidates = jobs.filter {
            $0.jobID != jobID
                && $0.kind == "video_performance"
                && $0.status == "succeeded"
                && $0.viewable
        }.prefix(12)
        for candidate in candidates {
            guard isCurrentDiscovery(epoch: epoch, requestID: requestID) else { return }
            await loadReviewBundle(jobID: candidate.jobID)
        }
    }

    func compatibleReviewJobs(for jobID: String) -> [JobSummary] {
        guard let reference = reviewBundles[jobID] else { return [] }
        return jobs.filter { candidate in
            guard candidate.jobID != jobID,
                  let bundle = reviewBundles[candidate.jobID]
            else { return false }
            return bundle.comparisonKey == reference.comparisonKey
        }
    }

    private func runtimeDidBecomeReady(endpoint: LoopbackEndpoint, token: String) {
        guard supervisor?.endpoint == endpoint, supervisor?.token == token else { return }
        invalidateAsyncOperations()
        clearRuntimeData()
        refresh()
    }

    private func runtimeDidStop() {
        invalidateAsyncOperations()
        clearRuntimeData()
    }

    private func invalidateAsyncOperations() {
        runtimeEpoch &+= 1
        refreshTask?.cancel()
        refreshTask = nil
        activeRefreshRequestID = nil
        refreshing = false
        cancelReviewOperations()
    }

    private func cancelReviewOperations() {
        reviewDiscoveryTask?.cancel()
        reviewDiscoveryTask = nil
        activeDiscoveryRequestID = nil
        for task in reviewBundleTasks.values {
            task.cancel()
        }
        reviewBundleTasks.removeAll()
        reviewBundleRequestIDs.removeAll()
        loadingReviewJobIDs.removeAll()
    }

    private func clearRuntimeData() {
        health = nil
        jobs = []
        reviewBundles = [:]
        reviewBundleErrors = [:]
        loadingReviewJobIDs = []
        requestError = nil
    }

    private func allocateRequestID() -> UInt64 {
        nextRequestID &+= 1
        return nextRequestID
    }

    private func isCurrentRuntime(
        epoch: UInt64,
        endpoint: LoopbackEndpoint,
        token: String
    ) -> Bool {
        runtimeEpoch == epoch
            && supervisor?.endpoint == endpoint
            && supervisor?.token == token
    }

    private func isCurrentDiscovery(epoch: UInt64, requestID: UInt64) -> Bool {
        !Task.isCancelled
            && runtimeEpoch == epoch
            && activeDiscoveryRequestID == requestID
    }

    private func finishRefresh(requestID: UInt64) {
        guard activeRefreshRequestID == requestID else { return }
        activeRefreshRequestID = nil
        refreshTask = nil
        refreshing = false
    }

    private func finishReviewBundleLoad(jobID: String, requestID: UInt64) {
        guard reviewBundleRequestIDs[jobID] == requestID else { return }
        reviewBundleRequestIDs.removeValue(forKey: jobID)
        reviewBundleTasks.removeValue(forKey: jobID)
        loadingReviewJobIDs.remove(jobID)
    }

    private static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        return (error as? URLError)?.code == .cancelled
    }
}

enum LibrarySection: String, CaseIterable, Identifiable {
    case jobs = "Recent Jobs"
    case diagnostics = "Diagnostics"

    var id: String { rawValue }
}
