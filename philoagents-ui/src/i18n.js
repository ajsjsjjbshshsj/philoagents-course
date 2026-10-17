
const messages = {
  zh: {
    play: "开始游戏",
    instructions: "操作说明",
    support: "支持原项目",
    language: "语言：中文（点击切换）",
    move: "方向键：移动人物",
    talk: "空格键：与哲学家交谈",
    closeDialogue: "ESC：关闭对话",
    close: "关闭",
    input: "输入中文或英文，回车发送",
    paused: "游戏已暂停",
    resume: "继续游戏",
    mainMenu: "返回主菜单",
    reset: "重新开始",
    resetFailed: "重置游戏失败，请重试",
  },
  en: {
    play: "Let's Play!",
    instructions: "Instructions",
    support: "Support PhiloAgents",
    language: "Language: English (Switch)",
    move: "Arrow keys: Move",
    talk: "SPACE: Talk to philosophers",
    closeDialogue: "ESC: Close dialogue",
    close: "Close",
    input: "Type your message and press Enter",
    paused: "GAME PAUSED",
    resume: "Resume Game",
    mainMenu: "Main Menu",
    reset: "Reset Game",
    resetFailed: "Failed to reset game. Try again.",
  }
};

export function t(key) {
  const lang = localStorage.getItem("game_language") || "zh";
  return messages[lang]?.[key] || key;
}

export function toggleLanguage() {
  const lang = localStorage.getItem("game_language") || "zh";
  localStorage.setItem(
    "game_language",
    lang === "zh" ? "en" : "zh"
  );
}
