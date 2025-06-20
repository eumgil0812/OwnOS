---
title: "Print Processors"
datePublished: Fri Jun 20 2025 11:55:24 GMT+0000 (Coordinated Universal Time)
cuid: cmc4r6p51000102js5auhcl0o
slug: print-processors
tags: boot-or-logon-autostart-execution

---

#**Boot or Logon Autostart Execution**

## 🎯 이 공격은 뭐야?

> **프린터 드라이버 시스템을 낚아채서 악성코드를 SYSTEM 권한으로 실행시키는 테크닉**이야.

---

## 🖨️ 프린터 프로세서가 뭐야?

* **Print Processor (프린트 프로세서)**는 Windows가 인쇄 작업을 처리할 때 사용하는 **DLL 파일**이야.
    
* 이건 `spoolsv.exe`라는 **프린터 스풀러 서비스**가 **부팅할 때 자동으로 로딩**해.
    

> 🎯 즉, 부팅만 하면 자동으로 내 DLL을 실행해주는 꿈의 자리✨

---

## 🧨 공격 방식은?

공격자는 아래 둘 중 하나로 **악성 DLL을 print processor로 등록**해.

### 1\. **API 이용 방법**

```bash
AddPrintProcessor("Windows x64", NULL, "evil.dll", "EvilProc");
```

* `SeLoadDriverPrivilege` 권한이 있으면 바로 가능해.
    

### 2\. **레지스트리 이용 방법**

```bash
[HKLM\SYSTEM\CurrentControlSet\Control\Print\Environments\Windows x64\Print Processors\EvilProc]
"Driver"="evil.dll"
```

* 여기에 등록하면 spooler가 부팅 시 자동으로 읽어줘.
    

---

## 📌 주의할 점은?

* 이 **evil.dll**은 반드시 **프린트 프로세서 디렉토리에 있어야 해.**
    
    * 경로는 `GetPrintProcessorDirectory()` API로 찾을 수 있어.
        
* 등록해놓고 나면 `spoolsv.exe` 재시작 or 재부팅 시 자동 실행!
    

---

## 👑 SYSTEM 권한? YES.

* 프린트 스풀러는 `SYSTEM` 권한으로 돌아가기 때문에,
    
* 여기에 붙은 **evil.dll도 SYSTEM 권한**으로 실행돼.
    

---

## 🎮 예를 들어

```bash
// 악성 DLL 내부
void RunDLL() {
  WinExec("powershell -enc ...", SW_HIDE);  // 백도어 실행
}
```

이런 식으로 백도어 실행, 권한 탈취, 혹은 커널 해킹까지도 가능해.

---

## 🧠 요약

| 포인트 | 설명 |
| --- | --- |
| 🎯 목적 | 시스템 부팅 시 악성 DLL 실행 + SYSTEM 권한 획득 |
| 🛠️ 위치 | 프린트 프로세서 등록 (API or 레지스트리) |
| 📦 필요 조건 | 악성 DLL이 정해진 디렉토리에 있어야 함 |
| 🔥 실행 시점 | `spoolsv.exe` 실행 시 (즉, 부팅 시) |
| 🧪 탐지 난이도 | 높음 (정상 프로세스에 숨겨짐) |