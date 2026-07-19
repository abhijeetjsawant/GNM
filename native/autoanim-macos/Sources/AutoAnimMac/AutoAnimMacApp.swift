import AppKit
import SwiftUI

final class AutoAnimAppDelegate: NSObject, NSApplicationDelegate {
    static var terminateRuntime: (() -> Void)?

    func applicationWillTerminate(_ notification: Notification) {
        Self.terminateRuntime?()
    }
}

@main
struct AutoAnimMacApp: App {
    @NSApplicationDelegateAdaptor(AutoAnimAppDelegate.self) private var appDelegate
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
                .task {
                    AutoAnimAppDelegate.terminateRuntime = { [weak model] in model?.stop() }
                    model.start()
                }
        }
        .defaultSize(width: 1_320, height: 820)
        .commands {
            CommandGroup(after: .appInfo) {
                Divider()
                Button("Restart Source Runtime") { model.restart() }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
            }
        }
    }
}
