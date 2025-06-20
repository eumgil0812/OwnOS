---
title: "Component Object Model"
datePublished: Fri Jun 20 2025 14:08:06 GMT+0000 (Coordinated Universal Time)
cuid: cmc4vxce7000002l86ito8sr9
slug: component-object-model
tags: inter-process-communication

---

## 🎯 **COM (Component Object Model): 해커의 비밀 통로**

## 🧩 기본 개념 (쉽게)

윈도우 내부에는 **"수많은 조각의 레고 블록"** 같은 컴포넌트들이 돌아다닌다.  
이걸 서로 연결해주는 시스템이 바로 **COM**.

* "야 Word야, 이 표 좀 가져와봐"
    
* "야 Explorer야, 이 파일 리스트 좀 줘봐"
    
* "야 네트워크야, 이 URL에 연결해줘"
    

이런 식으로 **프로그램끼리 서로 기능 빌려쓰게 해주는 시스템**  
윈도우의 **거의 모든 시스템**이 내부적으로 COM 사용 중.  
(Explorer, IE, Office, PowerShell, Windows Shell 등등...)

---

# 💣 **그런데 해커는?**

해커는 이걸 보고 이렇게 생각한다:

> "어라? 이 레고 블록 연결 고리만 조작하면...  
> 내 코드도 시스템처럼 '정상적으로' 실행될 수 있겠는데?"

---

# ⚔ **실제 공격 예시들**

### ① COM Hijacking (가장 유명한 기술)

* **<mark>윈도우 레지스트리에 어떤 COM 객체가 어떤 DLL을 쓸지 적혀있다.</mark>**
    
* 내가 이 레지스트리 키를 바꿔서 **내 악성 DLL 경로**를 등록해버림.
    
* 이후 정상 앱이 해당 COM 객체 호출 → 내 악성 DLL이 대신 실행됨.
    

**실전 사례:**  
2017년 APT 공격 그룹이 **Outlook COM Hijacking**을 이용해  
피해자의 아웃룩이 실행될 때마다 백도어가 열리게 만든 사례 발생.

---

### ② Fileless Malware + COM

* 악성코드를 파일로 저장하지 않고, **메모리에서만 실행**
    
* PowerShell + COM 인터페이스 조합 많이 사용됨
    
* 탐지 어려움 (디스크에 흔적 거의 없음)
    

**실전 사례:**  
FIN7 그룹이 Excel 문서 안 DDE + PowerShell + COM 객체를 활용해  
피해자 네트워크 장악 → 수백억 피해 입힘 (POS 시스템 해킹)

---

### ③ Scheduled Task 생성 (Persistence)

* **특정 COM 객체에는 스케줄 작업 생성 기능 있음**
    
* 이를 악용해 지속적으로 악성코드 실행 예약 가능
    

**실전 사례:**  
랜섬웨어 그룹들이 **"schtasks COM 객체"** 악용 →  
피해 시스템 재부팅할 때마다 악성코드 재실행

---

### ④ 원격 공격: DCOM + COM

* DCOM (Distributed COM): 네트워크를 넘어서 COM 호출 가능
    
* 내부망 lateral movement (옆으로 퍼지기) 공격에서 자주 사용
    

**실전 사례:**  
2020년 유명 랜섬웨어 그룹이 내부망 lateral movement 위해  
DCOM과 WMI COM 인터페이스 활용 → 수백 대 서버 감염

---

# 🔍 **왜 해커들이 COM을 사랑하는가?**

| 이유 | 설명 |
| --- | --- |
| 💡 **정상 기능 위장** | 정상 윈도우 시스템 기능 사용 → 탐지 회피 |
| 🚪 **시스템 깊숙이 침투** | 레지스트리, 서비스, 스케줄, 권한승급 다양하게 활용 |
| 🧙 **다양한 언어 지원** | C, C++, Python, PowerShell, VBScript 등 |
| 📡 **원격까지 가능** | DCOM 통해 원격 시스템까지 조작 가능 |

---

**한마디로 말해서:**

> 📣 "COM은 윈도우 안의 **합법적인 뒷문**이다.  
> **정상처럼 보이면서 몰래 놀 수 있는 놀이터**."