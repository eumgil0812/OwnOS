---
title: "LSASS Driver"
datePublished: Fri Jun 20 2025 11:47:44 GMT+0000 (Coordinated Universal Time)
cuid: cmc4qwtr0001t02jma5ir1byc
slug: lsass-driver
tags: boot-or-logon-autostart-execution

---

#**Boot or Logon Autostart Execution**

### 🎯 해커가 `LSASS` 드라이버를 건드리는 이유?

`LSASS (Local Security Authority Subsystem Service)`는 윈도우 보안의 핵심 엔진!

* 로그인, 사용자 인증, 도메인 토큰, 권한 관리 등은 다 여기서 처리돼.
    
* 이걸 제어할 수 있으면? **관리자처럼 움직일 수 있어.**
    

그래서 해커는 `lsass.exe` 안에서 동작하는 드라이버(DLL)를 **바꿔치기하거나 끼워넣기** 하려고 해.

---

### 🧪 실제 활용 방법 예시

#### ✅ 1. **드라이버 하이재킹 (Hijack Execution Flow)**

* `LSASS`가 특정 DLL을 로드하는 걸 알고, 그 **경로에 악성 DLL**을 넣어.
    
* 예: `C:\Windows\System32\foo.dll` → 여기다가 자신이 만든 악성 DLL을 심는 거지.
    
* 시스템 부팅 후 자동으로 **악성 DLL이 LSASS 권한으로 실행됨!**
    

#### ✅ 2. **Startup 등록 대신 LSA 플러그인 등록**

Startup 등록 vs LSA 플러그인 등록

| 구분 | Startup 등록 | LSA 플러그인 등록 |
| --- | --- | --- |
| 📂 위치 | Startup 폴더, Run 키 (`HKCU/HKLM...`) | `HKLM\SYSTEM\CurrentControlSet\Control\Lsa` |
| ⚙️ 실행 시점 | 사용자가 로그인할 때 | 부팅 후 **로그인 전에**, 시스템 초기화 중 |
| 🔐 권한 | 사용자 권한 (보통 일반 계정) | **SYSTEM 권한** (가장 높은 권한) |
| 👀 탐지 | 보안 제품에 흔히 탐지됨 | 숨기기 쉬움, 보안 제품 우회 가능 |
| 🎯 목적 | 간단한 악성코드 실행 | **지속적이고 은밀한 루트킷/백도어 실행** |

* 레지스트리에 LSA 플러그인을 설정할 수 있어.
    
    ```bash
    [HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Lsa]
    "Authentication Packages"=multi_sz: "msv1_0", "evilpack"
    ```
    
* `evilpack.dll`을 미리 `System32`에 심어두면, 부팅할 때 **자동 로딩**돼.
    

#### ✅ 3. **Dumping LSASS**

* 악성 코드가 `lsass.exe`를 메모리에서 덤프하고, **암호 해시, Kerberos 티켓** 같은 인증 정보를 추출.
    
* 툴 예시: `Mimikatz`, `ProcDump`
    

---

### 🕶️ 실전에서는 이렇게 사용돼

> 🎯 예: APT 공격자가 시스템에 침투 → LSA 플러그인에 자신들의 `persistence.dll`을 등록  
> → 재부팅해도 살아남음 → 이 DLL이 매번 실행되면서 공격자가 C2 서버와 통신 시작

---

### 🧩 왜 위험할까?

* LSASS는 커널과 깊게 연결돼 있어.
    
* 이걸 악용하면 일반 백신이나 EDR도 탐지 못하거나, 삭제 실패할 수 있음.
    
* 특히 **도메인 컨트롤러**에서 당하면? 조직 전체 계정 다 털릴 수 있어.
    

---

### 🧯 탐지 & 방어는?

| 방법 | 설명 |
| --- | --- |
| 🧪 Event ID 4611 | 새로운 LSA 모듈이 로딩됐는지 확인 가능 |
| 🔍 Sysinternals Autoruns | LSA 관련 레지스트리 등록 확인 가능 |
| 🧼 정기적으로 `lsass.exe` 덤프 탐지 | 이상 DLL 또는 스니핑 행위 탐지 |
| 🛡️ Credential Guard | Windows 기능으로 LSASS 보호 |

---

### 📚 참고로 이런 기법은 MITRE ATT&CK에서…

* **T1547.009 - Boot or Logon Autostart Execution: Shortcut Modification**
    
* **T1003.001 - OS Credential Dumping: LSASS Memory**