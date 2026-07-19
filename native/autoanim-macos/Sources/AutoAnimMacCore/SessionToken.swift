import Foundation
import Security

public enum SessionToken {
    public static func generate(byteCount: Int = 32) throws -> String {
        guard byteCount >= 16 else { throw SessionTokenError.insufficientEntropy }
        var bytes = [UInt8](repeating: 0, count: byteCount)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            throw SessionTokenError.randomGenerationFailed
        }
        return bytes.map { String(format: "%02x", $0) }.joined()
    }

    public static func isValid(_ token: String) -> Bool {
        token.count >= 64 && token.count.isMultiple(of: 2) && token.allSatisfy {
            $0.isNumber || ("a"..."f").contains(String($0))
        }
    }
}

public enum SessionTokenError: Error, Equatable {
    case insufficientEntropy
    case randomGenerationFailed
}
