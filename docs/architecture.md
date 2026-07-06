%md
# Architecture

> TV VOC AI Platform Architecture

---

# 1. Purpose

TV VOC AI Platform의 목적은 고객 리뷰를 월 단위로 안정적으로 수집, 구조화, 분류, 분석하여 상품기획 의사결정을 지원하는 것이다.

이 플랫폼은 단순한 리뷰 태깅 시스템이 아니라 아래 질문에 답하기 위한 구조를 가진다.

* 고객이 어떤 경험을 했는가
* 그 경험은 어떤 카테고리와 주제로 구조화되는가
* 어떤 경험 요소가 만족도 Driver인가
* 신규 카테고리, 신규 주제, 희소 주제를 어떤 기준으로 운영에 반영할 것인가

최종 산출물은 Topic Classification 자체가 아니라 Experience Dataset, Driver 분석 결과, AI Insight, Dashboard Export이다.

---

# 2. Architecture Principles

## 2.1 Design Domain And Operation Domain Separation

플랫폼은 설계부와 운영부를 분리한다.

* 설계부: Taxonomy와 분류 기준을 만들고 바꾸는 영역
* 운영부: 월배치로 데이터를 처리하고 기존 기준을 적용하는 영역

운영부가 매월 전체 분류 기준을 다시 설계하지 않도록 하고, 설계 변경이 필요한 경우에만 설계부로 전달하는 구조를 기본으로 한다.

## 2.2 ML First, LLM Fallback

초기에는 LLM 사용 비중이 높을 수 있으나 최종 목표는 Hybrid Classification이다.

운영 우선순위는 아래와 같다.

* Rule-based classification
* ML/DL classification
* LLM fallback classification
* Human review for new or ambiguous cases

즉 LLM은 운영 전량 분류기가 아니라 신규 카테고리, 신규 주제, 저신뢰 케이스, 설명 생성에 집중한다.

## 2.3 Explainable And Reviewable

모든 주요 결과는 근거를 추적할 수 있어야 한다.

* 어떤 rule_profile이 적용되었는가
* 어떤 topic_pool이 기준이었는가
* 어떤 memo가 대표 사례였는가
* 어떤 케이스가 low-confidence였는가

## 2.4 Reproducible And Versioned

동일 입력은 동일 출력이 가능해야 하며, 모든 설계물과 운영 산출물은 버전 관리되어야 한다.

핵심 버전 축:

* taxonomy_version
* prompt_version
* model_version
* pipeline_version

---

# 3. High-Level Architecture

```text
Raw Review Data
    ↓
Monthly Ingest / Standardization
    ↓
Design Check
    ├─ 신규 카테고리 / 신규 주제 후보 있음 → Design Pipeline
    └─ 기존 기준으로 처리 가능 → Operation Pipeline
    ↓
Hybrid Classification
    ├─ Rule
    ├─ ML/DL
    ├─ LLM Fallback
    └─ Human Review Queue
    ↓
Experience Dataset
    ↓
Driver Analysis
    ↓
Evidence And AI Insight
    ↓
Dashboard / Planning Support
```

---

# 4. Domain Architecture

## 4.1 Design Domain

설계부는 분류 기준을 생성하고 갱신하는 영역이다.

주요 책임:

* 신규 카테고리 탐지
* category pattern seed 생성
* rule_profile 생성 및 갱신
* topic_pool 생성 및 갱신
* `기타` / 저신뢰 결과에서 신규 topic 후보 탐지
* 희소 topic 통합 또는 재배치 기준 수립
* taxonomy review queue 운영

주요 산출물:

* `category_rule_master`
* `topic_pool`
* `taxonomy_definition`
* `prompt_registry`
* `review_queue`
* `taxonomy_version_history`

## 4.2 Operation Domain

운영부는 월배치 실행 본체이다.

주요 책임:

* 월별 원천 데이터 적재
* 기존 taxonomy 기준 자동 분류
* ML/DL 고신뢰 결과 채택
* 저신뢰 / 미분류 / ambiguous 건의 LLM fallback
* `기타` 비중 계산
* 신규 topic 후보 감지
* 희소 topic 재배치
* 최종 결과 적재
* Driver 분석 / Insight 생성

주요 산출물:

* `classification_detail`
* `classification_summary`
* `driver_input`
* `weighted_corr`
* `weighted_regression`
* `driver_selection`
* `ai_insight`
* `dashboard_export`
* `pipeline_progress`
* `pipeline_failed`
* `pipeline_run_log`

---

# 5. Monthly Batch Alignment

월배치 자체의 세부 절차는 `operation.md`에서 관리한다. 이 문서에서는 아키텍처 관점의 경계만 정의한다.

월배치에서 설계부와 운영부가 연결되는 대표 분기:

* 신규 `cate_1_depth` 또는 `cate_2_depth` 유입
* 기존 taxonomy로 설명되지 않는 `기타` 증가
* low-confidence 비중 급증
* 희소 topic 정리 필요

이 경우 운영부는 설계부 큐를 생성하고, 설계부는 새 version의 rule_profile / topic_pool / taxonomy_definition을 publish한다.

---

# 6. Taxonomy Design Architecture

설계부는 아래 구성으로 동작한다.

## Step 1. Group Sampling

카테고리/감성 그룹별 대표 샘플을 추출한다.

현재 원칙:

* `memo_id` 기준 dedupe
* `year`, `country`, `brand_name`, `device_type`, memo length 다양성 반영
* rule_profile용 prompt sample은 대표성 있는 제한 수 사용

## Step 2. Category Pattern Seed

정적 패턴이 있으면 우선 사용하고, 신규 카테고리면 dynamic seed를 생성한다.

대표 출력:

* feature_hint_terms
* reason_signal_terms
* overall_sentiment_terms
* candidate_topic_labels

## Step 3. Rule Profile Generation

카테고리/감성별 overall 허용/차단 기준과 feature/reason 기준을 생성한다.

대표 출력:

* overall_allowed_rule
* overall_block_rule
* feature_hint_terms
* reason_signal_terms
* non_overall_examples

## Step 4. Topic Pool Generation

rule_profile을 바탕으로 topic_pool을 생성한다.

대표 출력:

* topic
* description
* representative_memos

## Step 5. Review And Publish

신규 카테고리 또는 신규 topic은 검토 후 taxonomy_version에 반영한다.

---

# 7. Classification Architecture

분류 엔진은 단일 모델이 아니라 라우터 기반 다중 엔진 구조를 권장한다.

```text
Input Memo
    ↓
Rule Router
    ├─ Match → Rule Output
    └─ No Match
          ↓
      ML/DL Classifier
          ├─ High Confidence → Accept
          └─ Low Confidence
                ↓
            LLM Fallback
                ├─ Confident → Accept
                └─ Ambiguous / New Pattern → Review Queue
```

핵심 설계 포인트:

* Rule과 ML이 처리 가능한 케이스는 운영비용이 낮다
* LLM은 고비용 구간에만 집중해야 한다
* review queue는 전수 검토가 아니라 새 기준이 필요한 케이스만 모으는 구조여야 한다

---

# 8. Model Strategy

## 8.1 Embedding Model For Retrieval And Similarity

추천 1순위 예시:

* `BAAI/bge-m3`

선정 이유:

* multilingual 대응
* 짧은 리뷰와 혼합 언어 표현에 강점
* retrieval, clustering, similarity 용도로 범용성이 높음

활용 위치:

* 신규 topic 후보 유사도 검색
* representative memo selection
* topic mapping candidate retrieval
* low-confidence classification support

대안 예시:

* `intfloat/multilingual-e5-large`

## 8.2 Supervised ML Baseline

운영 초기에는 단순하고 해석 가능한 baseline이 필요하다.

추천 baseline:

* TF-IDF + Logistic Regression

장점:

* 빠름
* 재현성 높음
* confidence 기반 routing에 활용 가능
* subgroup별 점검이 쉬움

활용 위치:

* 기존 taxonomy가 안정된 카테고리의 1차 분류

## 8.3 DL Classifier For Multilingual Topic Classification

데이터가 충분히 누적되면 multilingual encoder 기반 분류기가 적합하다.

추천 예시:

* `xlm-roberta-base` fine-tuned for topic classification

선정 이유:

* 다국가 리뷰에 적합
* 한국어/영어/혼합 표현 대응력 우수
* topic classification fine-tuning 사례가 많음

활용 위치:

* 고빈도 카테고리의 본 운영 분류기
* confidence score 기반 LLM fallback 라우팅

대안 예시:

* `xlm-roberta-large`
* `mDeBERTa-v3-base`

## 8.4 New Topic Discovery Model

신규 주제 탐지는 분류기와 다른 성격이다. supervised classifier보다 embedding + clustering 조합이 더 적합할 수 있다.

추천 예시:

* `BERTopic`
  * embedding model: `BAAI/bge-m3`
  * dimensionality reduction: `UMAP`
  * clustering: `HDBSCAN`
  * topic representation: c-TF-IDF + LLM labeling

선정 이유:

* `기타` 내부의 잠재 주제를 찾기 좋음
* topic 수를 사전 고정하지 않아도 됨
* 대표 표현과 대표 memo를 같이 확인 가능

활용 위치:

* `기타` 내부 신규 주제 탐지
* low-confidence cluster review
* monthly taxonomy expansion candidate discovery

## 8.5 LLM Role

LLM은 다음 영역에 집중한다.

* 신규 카테고리 rule_profile 생성
* topic_pool 생성
* low-confidence memo 분류
* topic labeling / explanation
* Driver / Insight 서술 생성

즉 LLM은 운영 전량 분류기가 아니라, 설계 지원 및 예외 처리 엔진으로 본다.

---

# 9. Data Layer

## 9.1 Source

* raw_review_table
* sentiment_source_table
* model_meta_table

## 9.2 Reference

* category_rule_table
* category_master_table
* taxonomy_definition_table
* prompt_registry_table

## 9.3 Output

* rule_profile
* clue_keyword
* topic_pool
* classification_detail
* classification_summary
* driver_input
* weighted_corr
* weighted_regression
* driver_selection
* ai_insight
* dashboard_export

## 9.4 Logs

* pipeline_progress
* pipeline_failed
* pipeline_run_log

---

# 10. Module Structure

권장 코드 구조:

```text
src/
  common/
    config_loader.py
    llm_client.py
    memo_id.py

  taxonomy/
    group_sampler.py
    prompt_builder.py
    category_pattern_generator.py
    rule_profile_generator.py
    rule_profile_writer.py
    topic_pool_generator.py
    topic_pool_writer.py
    new_topic_detector.py

  classification/
    rule_classifier.py
    ml_classifier.py
    dl_classifier.py
    llm_fallback_classifier.py
    confidence_router.py

  operation/
    monthly_ingest.py
    monthly_category_check.py
    monthly_quality_monitor.py
    etc_topic_rebalancer.py

  pipeline/
    run_taxonomy_design_refresh.py
    run_monthly_pipeline.py
```

---

# 11. Governance And Review

운영상 반드시 관리해야 할 항목:

* 신규 카테고리 유입 건수
* 신규 topic 후보 건수
* `기타` 비중 추이
* low-confidence 비중 추이
* LLM fallback 사용량
* taxonomy version 변경 이력
* category별 분류 품질

리뷰는 전수 검토가 아니라 queue 기반 최소 개입 원칙으로 운영한다.

---

# 12. Recommended Next Build Order

현재 구조 기준 다음 구현 우선순위:

1. `topic_pool_writer.py`
2. `rule_profile -> topic_pool` 단일 그룹 생성/저장 검증
3. `run_taxonomy_design_refresh.py`
4. `monthly_category_check.py`
5. `classification` 인터페이스 설계
6. `ML-first / LLM-fallback` router 구현
7. `new_topic_detector.py`
8. `run_monthly_pipeline.py`

---

# 13. Final Direction

이 플랫폼의 최종 목표는 아래와 같다.

* 설계부는 taxonomy를 진화시킨다
* 운영부는 월배치로 대부분을 자동 처리한다
* ML/DL이 기본 분류를 담당한다
* LLM은 신규성, 모호성, 설명 생성에 집중한다
* 사람은 모든 데이터를 보지 않고, 바꿔야 할 기준만 검토한다

따라서 이 아키텍처는 현재의 LLM 기반 구현을 수용하면서도, 장기적으로 Hybrid AI Native 운영 플랫폼으로 확장 가능하도록 설계되어야 한다.
