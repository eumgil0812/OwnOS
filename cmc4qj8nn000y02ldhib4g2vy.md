---
title: "Boot or Logon Autostart Execution"
datePublished: Fri Jun 20 2025 11:37:10 GMT+0000 (Coordinated Universal Time)
cuid: cmc4qj8nn000y02ldhib4g2vy
slug: boot-or-logon-autostart-execution

---

💻 **"부팅 마법사: 해커의 자동 실행 주문"** 🎩✨

상상해봐!  
해커가 Windows 시스템에 숨어 들어왔어. 그런데 매번 몰래 실행 버튼을 누르기 귀찮잖아?  
그래서 뭐 하냐고?

바로 **자동 실행 마법**을 걸어버려! 😈  
이름하여 `Boot or Logon Autostart Execution` — 부팅하거나 로그인할 때 자동으로 실행되게 만드는 기술이지.

---

### 🧠 마법 1: **레지스트리 키 조작**

해커는 Windows 레지스트리에 요런 "비밀 스크롤"을 추가해.

* `HKEY_CURRENT_USER\...\Run`
    
* `HKEY_LOCAL_MACHINE\...\RunOnce`
    

이 키에 “어디에 있는 프로그램을 실행해라~” 하고 적어두면, **사용자가 로그인하자마자** 바로 실행돼.  
*예를 들면:*

```bash
reg add HKCU\...\Run /v NotReal /d "C:\Users\Skylar\evil.exe"
```

그럼 매번 로그인할 때마다 `evil.exe`가 자동으로 실행! 😱

---

### 🧙 마법 2: **시작 폴더에 심기**

Windows에는 **Startup Folder**라는 비밀 공간이 있어.  
여기에 프로그램을 몰래 넣으면? → 다음 로그인 때 바로 실행돼!

* `C:\Users\Skylar\AppData\Roaming\...\Startup` (개인용)
    
* `C:\ProgramData\...\Startup` (모든 사용자용)
    

---

### 🧟 마법 3: **서비스처럼 부활**

레지스트리의 `RunServices`, `RunServicesOnce` 같은 키를 사용하면,  
시스템 부팅될 때 서비스처럼 실행돼. 좀비처럼 다시 살아나는 거지!

---

### 🦹 보너스 스킬: **Masquerading (위장술)**

* 해커는 “evil.exe” 대신 “ChromeUpdate.exe” 같은 이름을 써서  
    관리자도 못 알아보게 숨겨둬.
    

---

### 💥 정말 무서운 점은?

💾 `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\BootExecute`  
여기에는 원래 디스크 체크 프로그램만 있어야 하는데...  
해커는 여기에 **자기 멀웨어**를 넣어. 그럼 **윈도우가 켜지기 전부터** 실행돼!! 😱😱