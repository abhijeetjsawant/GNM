import AutoAnimMacCore
import SwiftUI

struct ProductionProjectList: View {
    let projects: [ProductionProjectRecord]
    @Binding var selection: String?

    var body: some View {
        List(projects, selection: $selection) { project in
            VStack(alignment: .leading, spacing: 4) {
                Text(project.name).font(.headline)
                Text("\(project.shotCount) shot\(project.shotCount == 1 ? "" : "s") · \(project.lifecycle)")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .tag(project.id)
        }
        .overlay {
            if projects.isEmpty {
                ContentUnavailableView("No projects", systemImage: "film", description: Text("Create a project from the local API to begin organizing characters and shots."))
            }
        }
        .navigationTitle("Projects")
    }
}

struct ProductionCharacterList: View {
    let characters: [ProductionCharacterRecord]
    @Binding var selection: String?

    var body: some View {
        List(characters, selection: $selection) { character in
            VStack(alignment: .leading, spacing: 4) {
                Text(character.name).font(.headline)
                Text("Revision \(character.currentRevisionID) · \(character.consentStatus)")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .tag(character.id)
        }
        .overlay {
            if characters.isEmpty {
                ContentUnavailableView("No characters", systemImage: "person.crop.square", description: Text("Promote a fitted image or multiview reconstruction to create a reusable character."))
            }
        }
        .navigationTitle("Characters")
    }
}

struct ProductionShotList: View {
    let shots: [ProductionShotRecord]
    @Binding var selection: String?

    var body: some View {
        List(shots, selection: $selection) { shot in
            VStack(alignment: .leading, spacing: 4) {
                Text(shot.name).font(.headline)
                Text("\(shot.lifecycle) · \(shot.takeCount) take\(shot.takeCount == 1 ? "" : "s") · character \(shot.characterRevision.characterID.prefix(8))")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .tag(shot.id)
        }
        .overlay {
            if shots.isEmpty {
                ContentUnavailableView("No shots", systemImage: "rectangle.stack", description: Text("Create a shot with an exact character revision to start a performance."))
            }
        }
        .navigationTitle("Shots")
    }
}

struct ProductionProjectDetail: View {
    let project: ProductionProjectRecord
    var body: some View {
        ProductionDetail(title: project.name, subtitle: project.description ?? "No project description", icon: "film.stack.fill") {
            DetailRow(label: "Shots", value: "\(project.shotCount)")
            DetailRow(label: "Lifecycle", value: project.lifecycle)
            DetailRow(label: "Updated", value: project.updatedAt)
        }
    }
}

struct ProductionCharacterDetail: View {
    let character: ProductionCharacterRecord
    var body: some View {
        ProductionDetail(title: character.name, subtitle: "Reusable pinned character revision", icon: "person.crop.square.fill") {
            DetailRow(label: "Revision", value: character.currentRevisionID)
            DetailRow(label: "Consent", value: "\(character.consentScope) · \(character.consentStatus)")
            DetailRow(label: "Appearance", value: character.appearanceStatus)
            DetailRow(label: "Material rights", value: character.materialRightsStatus)
            DetailRow(label: "Body", value: character.bodyStatus)
        }
    }
}

struct ProductionShotDetail: View {
    let shot: ProductionShotRecord
    @ObservedObject var model: AppModel
    @State private var attachError: String?
    @State private var attachingJobID: String?

    private var attachableJobs: [JobSummary] {
        model.jobs.filter {
            $0.status == "succeeded"
                && ($0.kind == "audio_animation" || $0.kind == "video_performance")
        }
    }

    var body: some View {
        ProductionDetail(title: shot.name, subtitle: shot.description ?? "Pinned character performance workspace", icon: "rectangle.stack.fill") {
            DetailRow(label: "Stage", value: shot.lifecycle)
            DetailRow(label: "Pinned revision", value: shot.characterRevision.revisionID)
            DetailRow(label: "Takes", value: "\(shot.takeCount)")
            DetailRow(label: "Versions", value: "\(shot.versionCount)")
            DetailRow(label: "Latest version", value: shot.latestVersionID ?? "Not linked")
            if attachableJobs.isEmpty {
                DetailRow(label: "Performance", value: "No completed audio or video jobs")
            } else {
                Menu {
                    ForEach(attachableJobs) { job in
                        Button("Attach \(job.kind == "audio_animation" ? "audio" : "video") · \(job.input.name)") {
                            attachingJobID = job.id
                            attachError = nil
                            Task {
                                do {
                                    try await model.attachCompletedJob(job, to: shot)
                                } catch {
                                    attachError = error.localizedDescription
                                }
                                attachingJobID = nil
                            }
                        }
                    }
                } label: {
                    HStack {
                        Text("Attach completed performance")
                        Spacer()
                        if attachingJobID != nil { ProgressView() }
                    }
                }
                .disabled(attachingJobID != nil)
                .padding(12)
            }
            if let attachError {
                Text(attachError).font(.caption).foregroundStyle(.red).padding(.horizontal, 12)
            }
        }
    }
}

private struct ProductionDetail<Rows: View>: View {
    let title: String
    let subtitle: String
    let icon: String
    @ViewBuilder let rows: () -> Rows
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Label(title, systemImage: icon).font(.title2.weight(.semibold))
                Text(subtitle).foregroundStyle(.secondary)
                VStack(spacing: 0) { rows() }
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
                Text("This record is local, durable, and pinned. Performance jobs can be attached only when their source media and character revision match.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(24).frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

private struct DetailRow: View {
    let label: String
    let value: String
    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.caption.monospaced()).multilineTextAlignment(.trailing).textSelection(.enabled)
        }.padding(12)
    }
}
