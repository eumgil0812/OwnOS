---
title: "Registry Run Keys / Startup Folder"
datePublished: Fri Jun 20 2025 11:40:38 GMT+0000 (Coordinated Universal Time)
cuid: cmc4qnpef001g02l93ma6hbgu
slug: registry-run-keys-startup-folder
tags: boot-or-logon-autostart-execution

---

#**Boot or Logon Autostart Execution**

### 🎯 공격자가 레지스트리 또는 시작 폴더를 통해 지속성을 확보하는 방법

공격자는 **레지스트리의 실행(run) 키** 또는 **시작 프로그램 폴더(startup folder)** 에 악성 프로그램을 등록해, 사용자가 로그인할 때 자동 실행되도록 설정함으로써 **지속성(persistence)** 을 확보할 수 있습니다.

이렇게 등록된 프로그램은 해당 **사용자의 권한 수준**으로 실행됩니다.

---

### 🗂️ 윈도우 기본 실행(run) 키 (사용자 로그인 시 실행)

* `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run`
    
* `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\RunOnce`
    
* `HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Run`
    
* `HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\RunOnce`
    

💡 참고:

* `RunOnceEx` 키도 존재하지만 Vista 이후 버전에서는 기본적으로 생성되지 않음
    
* `RunOnceEx`의 `Depend` 값을 이용해 DLL을 로그인 시 자동 로딩하는 것도 가능함:
    
    ```bash
    reg add HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnceEx\0001\Depend /v 1 /d "C:\temp\evil.dll"
    ```
    

---

### 📁 시작 프로그램 폴더 경로

프로그램을 이 폴더에 넣으면 로그인 시 자동으로 실행됩니다.

* **개별 사용자용**:  
    `C:\Users\[사용자명]\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup`
    
* **모든 사용자용**:  
    `C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Startup`
    

---

### 🧩 시작 폴더 관련 레지스트리 키

다음 키들은 시작 폴더의 경로 또는 실행 항목을 지정할 수 있는 곳입니다:

* `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders`
    
* `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders`
    
* `HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders`
    
* `HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders`
    

---

### 🔧 서비스 자동 시작 관련 키 (부팅 시)

서비스 형태로 자동 실행되도록 설정할 수 있는 레지스트리 키:

* `HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\RunServicesOnce`
    
* `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\RunServicesOnce`
    
* `HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\RunServices`
    
* `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\RunServices`
    

---

### 🛠️ 그룹 정책을 통한 실행 등록

정책을 통해 시작 프로그램을 설정할 경우 다음 경로에 값이 생성됩니다:

* `HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Policies\Explorer\Run`
    
* `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Policies\Explorer\Run`
    

---

### 🧠 특이한 실행 위치들

* `HKEY_CURRENT_USER\Software\Microsoft\Windows NT\CurrentVersion\Windows`  
    → `load` 값에 등록된 프로그램이 로그인한 사용자 컨텍스트에서 자동 실행됨
    
* `HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager`  
    → `BootExecute` 값은 기본적으로 `autocheck autochk *` 로 되어 있는데,  
    여기에 악성 프로그램을 추가하면 **부팅 시 자동 실행**됨
    

---

### 🔥 요약

공격자는 이처럼 여러 **레지스트리 키**와 **시작 폴더**를 이용해 악성코드를 은밀하게 실행하고,  
부팅이나 재시작 후에도 시스템에 **지속적으로 접근 가능한 백도어**를 유지할 수 있습니다.

심지어 `Masquerading` 기법을 활용해 정식 프로그램처럼 위장할 수도 있습니다.  
예: `"AdobeUpdater.exe"` 처럼 보이도록 설정