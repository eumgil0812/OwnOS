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

* \*\*Print Processor (프린트 프로세서)\*\*는 Windows가 인쇄 작업을 처리할 때 사용하는 **DLL 파일**이야.
    
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

## 🎯 프린터 스풀러가 인기 있는 이유

### ✅ **1\. SYSTEM 권한으로 실행**

* 프린터 스풀러는 **Windows 부팅 시 자동으로 실행**되고,
    
* `SYSTEM` 권한을 가짐. → 공격자가 최고 권한에서 코드 실행 가능.
    

### ✅ **2\. 항상 실행되고 있다**

* 프린터 스풀러는 **대부분의 Windows에서 항상 켜져 있어**.
    
    * 개인용 PC, 서버, AD 도메인 컨트롤러까지 전부 포함.
        

### ✅ **3\. 너무 오래된 구조**

* Print Processor 같은 기능은 **1990년대부터 존재**해서,
    
* 구조가 복잡하고, 레거시 코드가 많고,
    
* **보안이 최신 설계에 비해 허술**해.
    

### ✅ **4\. 탐지 우회가 쉬움**

* spoolsv.exe는 정상 시스템 서비스라 의심이 적고,
    
* **DLL만 심어두면 자동 실행**되기 때문에 **행위 기반 탐지도 어렵다.**
    

### ✅ **5\. 관리자 권한이 아니어도 가능할 때가 있음**

* 예전 Windows에서는 **일반 사용자도 프린터 등록 가능**했음.
    
* 심지어 **원격으로도 공격 가능**했던 사례도 있었음 (e.g., PrintNightmare).
    

---

## 📌 실제 공격 예시: PrintNightmare (CVE-2021-34527)

* 이 취약점은 원격에서 print processor 등록이 가능해서,
    
* **네트워크 넘어 SYSTEM 권한 획득**까지 가능했어.
    
* 한동안 전 세계 윈도우 서버가 다 뚫릴 정도로 심각했지.
    

---

## 🧠 정리하자면

| 조건 | 프린터 스풀러의 특징 |
| --- | --- |
| 권한 | SYSTEM |
| 실행 시점 | 부팅 시 자동 |
| 사용 빈도 | 거의 모든 시스템에서 상시 실행 |
| 구조 | 오래되고 보안에 취약한 부분 많음 |
| 공격자 이득 | 쉬운 persistence, 쉬운 권한 상승 |