%md
# Operation

> VOC AI Platform Operational Workflow

---

# 1. Purpose

본 문서는 VOC AI Platform의 운영 프로세스를 정의한다.

프로젝트는 크게 두 단계로 운영된다.

1. Historical Backfill (초기 구축)
2. Monthly Batch Operation (정기 운영)

모든 AI Workflow는 동일한 Framework를 사용하며,

초기 구축 이후에는 신규 VOC만 분석 대상으로 처리한다.

---

# 2. Overall Operation Flow

```text
Historical Backfill

↓

Experience Dataset 구축

↓

Dashboard 초기 생성

↓

Monthly Batch

↓

Experience Dataset Update

↓

Driver Update

↓

Dashboard Refresh
```

---

# 3. Historical Backfill

최초 구축 단계

목적

과거 수년간의 VOC를 분석하여 플랫폼의 기준 데이터를 구축한다.

Input

* Historical VOC

Process

* Experience Taxonomy 구축
* Topic Classification
* Experience Dataset 생성
* Driver Analysis
* Dashboard 초기 구축

Output

* Taxonomy Version 1
* Experience Dataset
* Driver Dataset
* Dashboard

---

# 4. Monthly Batch Operation

매월 신규 VOC 입수 시 수행한다.

Input

* Monthly VOC

Process

① 신규 VOC 적재

↓

② 신규 Review 식별

↓

③ Topic Classification

↓

④ Experience Dataset Update

↓

⑤ Driver Analysis Refresh

↓

⑥ Dashboard Refresh

↓

⑦ AI Insight Refresh

Output

* Updated Experience Dataset
* Updated Driver Dataset
* Updated Dashboard

---

# 5. Operation Principles

## Incremental Processing

기존 데이터를 매월 다시 분석하지 않는다.

신규 VOC만 분석 대상으로 처리한다.

---

## Reproducible Analysis

동일 데이터는 동일한 결과를 생성해야 한다.

Taxonomy Version과 분석 Version을 함께 관리한다.

---

## Version Management

관리 대상

* Taxonomy Version
* Classification Version
* Driver Version

Version 변경 시 변경 이력을 관리한다.

---

# 6. Taxonomy Lifecycle

Taxonomy는 월 단위로 변경하지 않는다.

운영 단계

```text
Draft

↓

DS Review

↓

Production

↓

Monitoring

↓

Revision Candidate

↓

Next Version
```

신규 VOC에서 기존 Taxonomy로 설명하기 어려운 패턴이 지속적으로 발생하는 경우

다음 Version에서 반영한다.

---

# 7. Validation Cycle

운영 중 지속적으로 품질을 점검한다.

Validation 대상

* Topic Distribution
* Confidence Distribution
* Low Confidence Review
* Emerging Topic
* Error Pattern

Validation 결과는 다음 Taxonomy 개선 시 활용한다.

---

# 8. Dashboard Refresh

매월 Batch 완료 후 Dashboard를 갱신한다.

갱신 대상

* Topic Summary
* Driver Analysis
* Insight
* Trend Analysis

Dashboard는 항상 최신 Experience Dataset을 기준으로 생성한다.

---

# 9. Future Extension

현재 운영 대상

* VOC

향후 확장 가능한 데이터

* PRM
* Sales
* Pricing
* Competitor Reviews
* Product Specification

Multi-source Customer Experience Platform으로 확장 가능하도록 운영 구조를 유지한다.

---

# 10. Operational Principles

VOC AI Platform은 일회성 분석 프로젝트가 아니다.

매월 반복 수행되는 AI 분석 플랫폼을 목표로 한다.

운영 원칙

* Incremental Processing
* Stable Taxonomy
* Version Management
* Continuous Validation
* Reproducible Analysis
* AI-assisted Decision Support
