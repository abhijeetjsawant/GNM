import AutoAnimMacCore
import Foundation

@MainActor
final class AppModel: ObservableObject {
    @Published private(set) var configuration: RuntimeConfiguration?
    @Published private(set) var configurationError: String?
    @Published private(set) var health: HealthReport?
    @Published private(set) var jobs: [JobSummary] = []
    @Published var selectedJobID: String?
    @Published var selectedSection: LibrarySection = .jobs
    @Published private(set) var requestError: String?
    @Published private(set) var refreshing = false

    let supervisor: BackendSupervisor?

    init(bundle: Bundle = .main) {
        do {
            let configuration = try RuntimeConfiguration.from(bundle: bundle)
            self.configuration = configuration
            let supervisor = BackendSupervisor(configuration: configuration)
            self.supervisor = supervisor
            supervisor.onReady = { [weak self] _, _ in self?.refresh() }
            supervisor.onStopped = { [weak self] in
                self?.health = nil
                self?.jobs = []
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
        supervisor?.stop()
    }

    func restart() {
        health = nil
        jobs = []
        requestError = nil
        supervisor?.restart()
    }

    func refresh() {
        guard let endpoint = supervisor?.endpoint, let token = supervisor?.token else { return }
        refreshing = true
        requestError = nil
        let client = BackendClient(endpoint: endpoint, token: token)
        Task {
            do {
                async let fetchedHealth = client.health()
                async let fetchedJobs = client.recentJobs()
                let (newHealth, newJobs) = try await (fetchedHealth, fetchedJobs)
                health = newHealth
                jobs = newJobs
                if selectedJobID == nil || !newJobs.contains(where: { $0.jobID == selectedJobID }) {
                    selectedJobID = newJobs.first?.jobID
                }
            } catch {
                requestError = error.localizedDescription
            }
            refreshing = false
        }
    }
}

enum LibrarySection: String, CaseIterable, Identifiable {
    case jobs = "Recent Jobs"
    case diagnostics = "Diagnostics"

    var id: String { rawValue }
}
