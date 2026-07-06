%md
# Operation

> TV VOC AI Platform Operational Workflow

---

# 1. Purpose

본 문서는 TV VOC AI Platform의 운영 프로세스를 정의한다.

운영은 설계부와 분리된 별도 도메인으로 관리한다. 설계부가 분류 기준을 생성하고 버전화하면, 운영부는 그 기준을 월배치로 적용하고 성능을 모니터링한다.

프로젝트는 크게 두 단계로 운영된다.

1. Historical Backfill (초기 구축)
2. Monthly Batch Operation (정기 운영)

초기 구축 이후에는 신규 VOC만 분석 대상으로 처리하는 것을 원칙으로 한다.

---

# 2. Overall Operation Flow

```text
Historical Backfill
    ↓
Taxonomy Version 1 Build
    ↓
Experience Dataset Build
    ↓
Dashboard Initial Release
    ↓
Monthly Batch
    ↓
Incremental VOC Ingest
    ↓
Hybrid Classification
    ↓
Experience Dataset Update
    ↓
Driver / Insight Refresh
    ↓
Dashboard Refresh
```

---

# 3. Historical Backfill

최초 구축 단계

목적

과거 수년간의 VOC를 분석하여 플랫폼의 기준 데이터를 구축한다. 이 단계는 운영부가 사용할 초도 taxonomy, 초도 classifier baseline, 초도 experience dataset을 만드는 작업이다.

Input

* Historical VOC

Process

* Experience Taxonomy 구축
* Rule Profile / Topic Pool 생성
* Topic Classification
* Experience Dataset 생성
* Driver Analysis
* Dashboard 초기 구축

Output

* Taxonomy Version 1
* Experience Dataset
* Driver Dataset
* Dashboard Initial Release

---

# 4. Monthly Batch Operation

매월 신규 VOC 입수 시 수행한다.

Input

* Monthly VOC
* Latest Production Taxonomy
* Latest Classification Models

Process

① 신규 VOC 적재 및 표준화

↓

② 신규 Review 식별 및 Incremental 대상 확정

↓

③ 신규 카테고리 / 신규 주제 후보 여부 점검

↓

④ Hybrid Topic Classification

↓

⑤ `기타` / Low Confidence / Sparse Topic 재평가

↓

⑥ Experience Dataset Update

↓

⑦ Driver Analysis Refresh

↓

⑧ AI Insight Refresh

↓

⑨ Dashboard Refresh

Output

* Updated Experience Dataset
* Updated Driver Dataset
* Updated Dashboard
* Monthly Quality Metrics
* Review Queue Candidates

---

# 5. Operation Principles

## Incremental Processing

기존 데이터를 매월 다시 분석하지 않는다.

신규 VOC만 분석 대상으로 처리한다.

---

## Stable Production Taxonomy

운영 단계에서는 Production으로 승인된 taxonomy_version만 사용한다.

신규 카테고리 또는 신규 topic 후보가 발견되더라도, 설계부 검토 없이 즉시 Production Taxonomy를 바꾸지 않는다.

---

## Reproducible Analysis

동일 데이터는 동일한 결과를 생성해야 한다.

Taxonomy Version과 분석 Version을 함께 관리한다.

---

## Version Management

관리 대상

* Taxonomy Version
* Classification Version
* Model Version
* Driver Version
* Pipeline Version

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

Staging

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

# 7. Hybrid Classification In Operation

운영부의 기본 분류 전략은 Hybrid 구조이다.

우선순위:

1. Rule-based classification
2. ML/DL classification
3. LLM fallback classification
4. Review queue

운영 원칙:

* 명확한 패턴은 Rule로 처리
* 고신뢰 분류는 ML/DL 결과를 우선 채택
* 저신뢰 또는 신규 패턴은 LLM fallback 사용
* LLM 결과도 즉시 taxonomy 변경으로 연결하지 않고 review queue를 거친다

---

# 8. Validation Cycle

운영 중 지속적으로 품질을 점검한다.

Validation 대상

* Topic Distribution
* Confidence Distribution
* Low Confidence Review
* Emerging Topic
* Error Pattern
* `기타` 비중
* Sparse Topic 비중
* LLM fallback 비중

Validation 결과는 다음 Taxonomy 개선 시 활용한다.

---

# 9. Review Triggers

운영 중 아래 조건이 발생하면 설계부 또는 검토 큐로 전달한다.

* 신규 `cate_1_depth` 또는 `cate_2_depth` 유입
* `기타` 내부 표현군 비중이 임계치 이상
* 특정 topic 비중이 너무 낮아 운영상 유지 의미가 낮음
* low-confidence 비중 급증
* 모델 성능 드리프트 감지

이 문서에서는 trigger만 정의하고, 세부 재설계 방식은 설계부 문서와 코드에서 관리한다.

---

# 10. Dashboard Refresh

매월 Batch 완료 후 Dashboard를 갱신한다.

갱신 대상

* Topic Summary
* Driver Analysis
* Insight
* Trend Analysis

Dashboard는 항상 최신 Experience Dataset을 기준으로 생성한다.

---

# 11. Failure And Retry Policy

운영 배치는 실패 복구 가능해야 한다.

원칙:

* stage 단위 재실행 가능
* progress / failed log 저장
* partial success 여부 기록
* 동일 run_id 기준 재현 가능

권장 로그:

* pipeline_progress
* pipeline_failed
* pipeline_run_log

---

# 12. Future Extension

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

# 13. Operational Summary

VOC AI Platform은 일회성 분석 프로젝트가 아니다.

매월 반복 수행되는 AI 분석 플랫폼을 목표로 한다.

운영 원칙

* Incremental Processing
* Stable Taxonomy
* Version Management
* Continuous Validation
* Reproducible Analysis
* Hybrid AI Routing
* AI-assisted Decision Support
