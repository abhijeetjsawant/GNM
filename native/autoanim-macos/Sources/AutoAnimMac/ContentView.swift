import AutoAnimMacCore
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @State private var showingNewProject = false
    @State private var showingNewShot = false

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 180, ideal: 220, max: 280)
        } content: {
            content
                .navigationSplitViewColumnWidth(min: 260, ideal: 340, max: 460)
        } detail: {
            detail
        }
        .frame(minWidth: 1_040, minHeight: 680)
        .toolbar {
            ToolbarItemGroup {
                Button {
                    model.refresh()
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .disabled(model.supervisor?.endpoint == nil || model.refreshing)
                .keyboardShortcut("r", modifiers: .command)

                Button {
                    model.restart()
                } label: {
                    Label("Restart Runtime", systemImage: "restart")
                }
                .disabled(model.supervisor == nil)

                Button {
                    showingNewProject = true
                } label: {
                    Label("New Project", systemImage: "plus.rectangle.on.folder")
                }
                .disabled(model.supervisor?.endpoint == nil)

                Button {
                    showingNewShot = true
                } label: {
                    Label("New Shot", systemImage: "plus.rectangle.on.rectangle")
                }
                .disabled(model.supervisor?.endpoint == nil || model.selectedProject == nil || model.productionLibrary.characters.isEmpty)
            }
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            sourceRuntimeBanner
        }
        .sheet(isPresented: $showingNewProject) {
            NewProjectSheet(model: model)
        }
        .sheet(isPresented: $showingNewShot) {
            if let project = model.selectedProject {
                NewShotSheet(model: model, project: project)
            }
        }
    }

    private var sidebar: some View {
        List(LibrarySection.allCases, selection: $model.selectedSection) { section in
            Label(
                section.rawValue,
                systemImage: section.systemImage
            )
            .tag(section)
        }
        .navigationTitle("AutoAnim")
        .safeAreaInset(edge: .bottom) {
            VStack(alignment: .leading, spacing: 5) {
                Label(runtimeStateLabel, systemImage: runtimeStateSymbol)
                    .font(.caption.weight(.semibold))
                if let health = model.health {
                    Text("Pipeline health: \(health.status)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(.bar)
        }
    }

    @ViewBuilder
    private var content: some View {
        switch model.selectedSection {
        case .projects:
            ProductionProjectList(
                projects: model.productionLibrary.projects,
                selection: $model.selectedProjectID
            )
        case .characters:
            ProductionCharacterList(
                characters: model.productionLibrary.characters,
                selection: $model.selectedCharacterID
            )
        case .shots:
            ProductionShotList(
                shots: model.productionLibrary.shots,
                selection: $model.selectedShotID
            )
        case .jobs:
            jobsList
        case .diagnostics:
            DiagnosticsView(model: model)
        }
    }

    private var jobsList: some View {
        List(selection: $model.selectedJobID) {
            if model.jobs.isEmpty {
                ContentUnavailableView(
                    "No jobs available",
                    systemImage: "cube.transparent",
                    description: Text(emptyJobsDescription)
                )
                .listRowSeparator(.hidden)
            } else {
                ForEach(model.jobs) { job in
                    JobRow(job: job)
                        .tag(job.jobID)
                }
            }
        }
        .navigationTitle("Recent Jobs")
        .overlay(alignment: .bottom) {
            if let error = model.requestError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(10)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                    .padding()
            }
        }
    }

    @ViewBuilder
    private var detail: some View {
        switch model.selectedSection {
        case .projects:
            if let project = model.selectedProject {
                ProductionProjectDetail(project: project)
            } else {
                productionSelectionEmpty("Select a project", "film")
            }
        case .characters:
            if let character = model.selectedCharacter {
                ProductionCharacterDetail(character: character)
            } else {
                productionSelectionEmpty("Select a character", "person.crop.square")
            }
        case .shots:
            if let shot = model.selectedShot {
                ProductionShotDetail(shot: shot, model: model)
            } else {
                productionSelectionEmpty("Select a shot", "rectangle.stack")
            }
        case .jobs:
            if let job = model.selectedJob {
                JobDetail(job: job, model: model)
            } else {
                productionSelectionEmpty("Select a job", "viewfinder")
            }
        case .diagnostics:
            DiagnosticsDetail(model: model)
        }
    }

    private func productionSelectionEmpty(_ title: String, _ icon: String) -> some View {
        ContentUnavailableView(title, systemImage: icon, description: Text("The local production library will appear when the runtime has records."))
    }

    private var sourceRuntimeBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "hammer.fill")
            Text("Development build · uses this checkout’s Python runtime and assets")
                .font(.caption.weight(.semibold))
            Spacer()
            if let revision = Bundle.main.object(forInfoDictionaryKey: "AutoAnimSourceRevision") as? String,
               !revision.hasPrefix("__") {
                Text(String(revision.prefix(12)))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 7)
        .background(Color.orange.opacity(0.15))
        .overlay(alignment: .bottom) { Divider() }
    }

    private var runtimeStateLabel: String {
        if let error = model.configurationError { return "Configuration failed: \(error)" }
        return model.supervisor?.state.label ?? "Unavailable"
    }

    private var runtimeStateSymbol: String {
        switch model.supervisor?.state {
        case .ready: return "checkmark.circle.fill"
        case .starting: return "hourglass"
        case .failed: return "exclamationmark.triangle.fill"
        default: return "stop.circle"
        }
    }

    private var emptyJobsDescription: String {
        switch model.supervisor?.state {
        case .starting: return "The authenticated source runtime is starting."
        case .failed(let error): return error
        case .ready: return "The current source artifact store has no jobs."
        default: return model.configurationError ?? "Start or restart the source runtime."
        }
    }
}

private struct NewProjectSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var description = ""
    @State private var submitting = false
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("New Project").font(.title2.weight(.semibold))
            Text("Projects are the durable home for characters, shots and delivery work.")
                .foregroundStyle(.secondary)
            TextField("Project name", text: $name)
            TextField("Description (optional)", text: $description, axis: .vertical)
                .lineLimit(3...5)
            if let error { Text(error).font(.caption).foregroundStyle(.red) }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.disabled(submitting)
                Button("Create") {
                    submitting = true
                    error = nil
                    Task {
                        do {
                            try await model.createProject(name: name, description: description.nilIfBlank)
                            dismiss()
                        } catch {
                            self.error = error.localizedDescription
                            submitting = false
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || submitting)
            }
        }
        .padding(24)
        .frame(width: 440)
    }
}

private struct NewShotSheet: View {
    @ObservedObject var model: AppModel
    let project: ProductionProjectRecord
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var description = ""
    @State private var characterID = ""
    @State private var submitting = false
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("New Shot").font(.title2.weight(.semibold))
            Text("Pins the selected character revision to \(project.name).")
                .foregroundStyle(.secondary)
            TextField("Shot name", text: $name)
            Picker("Character", selection: $characterID) {
                Text("Choose a character").tag("")
                ForEach(model.productionLibrary.characters) { character in
                    Text("\(character.name) · \(character.currentRevisionID.prefix(8))")
                        .tag(character.id)
                }
            }
            TextField("Description (optional)", text: $description, axis: .vertical)
                .lineLimit(3...5)
            if let error { Text(error).font(.caption).foregroundStyle(.red) }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.disabled(submitting)
                Button("Create") {
                    guard let character = model.productionLibrary.character(id: characterID) else { return }
                    submitting = true
                    error = nil
                    Task {
                        do {
                            try await model.createShot(
                                projectID: project.id,
                                name: name,
                                characterID: character.id,
                                characterRevisionID: character.currentRevisionID,
                                description: description.nilIfBlank
                            )
                            dismiss()
                        } catch {
                            self.error = error.localizedDescription
                            submitting = false
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || characterID.isEmpty || submitting)
            }
        }
        .padding(24)
        .frame(width: 440)
    }
}

private extension String {
    var nilIfBlank: String? {
        let trimmed = trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

private struct JobRow: View {
    let job: JobSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(job.input.name)
                    .font(.headline)
                    .lineLimit(1)
                Spacer()
                Image(systemName: job.viewable ? "cube.fill" : "doc.text")
                    .foregroundStyle(job.viewable ? .green : .secondary)
            }
            Text(job.kind.replacingOccurrences(of: "_", with: " "))
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Text(job.status.capitalized)
                if job.warningCount > 0 {
                    Text("· \(job.warningCount) warning\(job.warningCount == 1 ? "" : "s")")
                        .foregroundStyle(.orange)
                }
            }
            .font(.caption2)
        }
        .padding(.vertical, 5)
    }
}

private struct JobDetail: View {
    let job: JobSummary
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(job.input.name).font(.headline)
                    Text("\(job.kind) · \(job.status) · \(job.jobID)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(12)
            .background(.bar)
            Divider()
            if job.viewable,
               let endpoint = model.supervisor?.endpoint,
               let token = model.supervisor?.token {
                if job.kind == "video_performance" {
                    ReviewWorkspaceView(
                        job: job,
                        model: model,
                        endpoint: endpoint,
                        token: token
                    )
                    .id("review-\(endpoint.baseURL.absoluteString)-\(job.jobID)")
                } else {
                    AuthenticatedViewer(endpoint: endpoint, token: token, jobID: job.jobID)
                        .id("\(endpoint.baseURL.absoluteString)-\(job.jobID)")
                }
            } else {
                ContentUnavailableView(
                    "No viewable 3D artifact",
                    systemImage: "cube.transparent",
                    description: Text("This job can still be inspected in Diagnostics and its sealed manifest.")
                )
            }
        }
    }
}

private struct DiagnosticsView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        List {
            Section("Source runtime") {
                ForEach(model.diagnostics) { item in
                    VStack(alignment: .leading, spacing: 3) {
                        Label(item.label, systemImage: item.ready ? "checkmark.circle.fill" : "xmark.circle.fill")
                            .foregroundStyle(item.ready ? .green : (item.required ? .red : .orange))
                        Text(item.path)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                }
            }
            if let health = model.health {
                Section("Pipeline health") {
                    ForEach(health.checks.keys.sorted(), id: \.self) { name in
                        if let check = health.checks[name] {
                            VStack(alignment: .leading, spacing: 3) {
                                Label(name, systemImage: check.ready ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                                    .foregroundStyle(check.ready ? .green : .orange)
                                Text(check.detail)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle("Diagnostics")
    }
}

private struct DiagnosticsDetail: View {
    @ObservedObject var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Runtime log").font(.title2.weight(.semibold))
                Text(model.supervisor?.logLines.joined(separator: "\n") ?? "No runtime log is available.")
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
                Text("This app is a development-signed source-runtime shell. It is not self-contained, notarized, sandboxed, or ready for distribution.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(24)
        }
    }
}
