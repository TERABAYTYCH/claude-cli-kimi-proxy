import { logger } from "../utils/logger.js";

const DELTA_SOFT_LIMIT = 60000; // chars

export interface ResumeDeltaDiag {
  sinceIndex?: number;
  messagesLen?: number;
  sliceLen?: number;
  sliceRoles?: string[];
}

/**
 * Detect structural regressions in a resumed CLI delta. A correct delta only
 * carries new user/tool messages; it must never replay assistant history or
 * the Kimi identity block, and it should stay small enough that resume savings
 * are real. Logs warnings but does not change request handling.
 */
export function checkResumeDeltaHealth(
  prompt: string,
  resume: boolean,
  sessionKey?: string,
  diag?: ResumeDeltaDiag
): void {
  if (!resume) return;

  // A correct delta never contains an assistant-turn block, because
  // openaiToCliDelta filters role !== "assistant". A literal "<previous_response>"
  // can appear inside a <tool_result> (e.g. source code the model read), so we
  // only flag it when it appears before any <tool_result> opening tag.
  const firstPrevResponse = prompt.indexOf("<previous_response>");
  const firstToolResult = prompt.indexOf("<tool_result>");
  const hasReplayedAssistant =
    firstPrevResponse !== -1 &&
    (firstToolResult === -1 || firstPrevResponse < firstToolResult);
  if (hasReplayedAssistant) {
    logger.warn("[Session] Broken resume delta: assistant history replayed in prompt", {
      sessionKey,
      promptChars: prompt.length,
      sinceIndex: diag?.sinceIndex,
      messagesLen: diag?.messagesLen,
    });
  }

  if (prompt.slice(0, 500).includes("You are Kimi Code CLI")) {
    logger.warn("[Session] Broken resume delta: Kimi identity leaked into prompt body", {
      sessionKey,
      promptChars: prompt.length,
      sinceIndex: diag?.sinceIndex,
      messagesLen: diag?.messagesLen,
    });
  }

  if (prompt.length > DELTA_SOFT_LIMIT) {
    logger.warn("[Session] Resume delta exceeds soft size limit", {
      sessionKey,
      promptChars: prompt.length,
      sinceIndex: diag?.sinceIndex,
      messagesLen: diag?.messagesLen,
      softLimit: DELTA_SOFT_LIMIT,
    });
  }
}
