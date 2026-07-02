%md
# Architecture

> VOC AI Platform Architecture Design

---

# 1. Project Mission

VOC AI Platform은 고객 리뷰를 AI 기반으로 구조화하여 상품기획 의사결정을 지원하는 플랫폼이다.

본 프로젝트의 목적은 리뷰를 단순 집계하는 것이 아니라 고객 경험(Customer Experience)을 데이터로 변환하여

* 고객이 무엇을 경험했는지
* 어떤 경험이 고객 만족을 결정하는지
* 무엇을 우선 개선해야 하는지

를 데이터 기반으로 설명하는 것이다.

---

# 2. Design Philosophy

VOC AI Platform은 단순한 LLM 기반 주제분류 시스템이 아니다.

프로젝트는 아래 원칙을 따른다.

## AI Native

AI는 반복 작업을 수행한다.

사람은 분석 Framework를 설계한다.

## Explainable

모든 분석 결과는

* 왜 그렇게 분류되었는가
* 어떤 리뷰가 근거인가

를 설명할 수 있어야 한다.

## Human in the Loop

사람은 모든 리뷰를 검토하지 않는다.

AI가 대부분의 리뷰를 처리하고,

불확실하거나 새로운 패턴만 사람이 검토한다.

## Reproducible

동일 데이터는 동일한 결과를 생성해야 한다.

모든 분석은 Version 관리가 가능해야 한다.

---

# 3. Overall Architecture

```text
Customer Reviews

↓

Customer Experience Structuring

↓

Experience Dataset

↓

Driver Analysis

↓

Evidence

↓

Insight

↓

Planning Support
```

Topic Classification은 최종 목적이 아니다.

Driver Analysis도 최종 목적이 아니다.

최종 목표는 상품기획 의사결정을 지원하는 것이다.

---

# 4. AI Workflow

## Step 1

Experience Taxonomy

고객 경험을 구조화하기 위한 Taxonomy를 구축한다.

구성

* Rule Profile
* Topic Pool
* Topic Definition
* Representative Reviews

Output

Taxonomy Catalog

---

## Step 2

Topic Classification

모든 리뷰를 Experience Taxonomy 기반으로 구조화한다.

현재

LLM Classification

향후

Rule

↓

Embedding Search

↓

LLM Validation

↓

Confidence

↓

Evidence

---

## Step 3

Experience Dataset

Topic Classification 결과를 분석 가능한 형태로 변환한다.

Experience Dataset은 플랫폼의 핵심 데이터이다.

예시

* Brand
* Country
* Product
* Category
* Topic
* Sentiment
* Confidence

---

## Step 4

Driver Analysis

Experience Dataset을 기반으로

고객 만족 Driver를 분석한다.

분석 방법

* Correlation
* Regression
* Feature Selection

Output

Driver Dataset

---

## Step 5

Evidence Generation

Driver 결과를 설명하는 근거를 생성한다.

예시

* 대표 리뷰
* 대표 키워드
* Confidence
* 시장별 차이

---

## Step 6

AI Insight

Driver와 Evidence를 기반으로

상품기획자가 활용 가능한

Insight를 생성한다.

---

# 5. Data Architecture

```text
Raw Review

↓

Standard Review

↓

Topic Classification

↓

Experience Dataset

↓

Driver Dataset

↓

Insight Dataset
```

Customer Experience Dataset은

모든 분석의 기준 데이터이다.

---

# 6. Core Components

## Taxonomy

Customer Experience를 정의한다.

---

## Classification

Review를 Experience Dataset으로 변환한다.

---

## Driver Analysis

핵심 경험 요인을 정량적으로 분석한다.

---

## Insight

Driver를 사람이 이해할 수 있는 형태로 변환한다.

---

## Dashboard

Experience Dataset과 Driver Dataset을 시각화한다.

---

## Planning Support

상품기획 의사결정을 지원한다.

---

# 7. Validation Framework

프로젝트는

분류 결과뿐 아니라

분류 품질도 관리한다.

Validation 대상

* Taxonomy Coverage
* Topic Representativeness
* Classification Accuracy
* Confidence
* Error Analysis

Validation 결과는

Taxonomy 개선에 지속적으로 반영된다.

---

# 8. Version Management

모든 핵심 자산은 Version 관리한다.

대상

* Taxonomy
* Classification
* Driver Model
* Insight

Version은

재현 가능한 분석을 위한 기준이 된다.

---

# 9. Future Architecture

향후 플랫폼은 다음 기능을 단계적으로 추가한다.

Phase 1

* Experience Taxonomy
* Topic Classification

Phase 2

* Hybrid Classification
* Embedding Search
* Confidence

Phase 3

* Driver Analysis
* AI Insight

Phase 4

* Multi-source Customer Experience Platform

Phase 5

* Planning Support Agent

---

# 10. Long-term Vision

VOC AI Platform은 리뷰 분석 시스템이 아니다.

고객 경험을 구조화하고,

고객 만족 Driver를 정량적으로 분석하며,

상품기획자가 데이터 기반으로 의사결정을 수행할 수 있도록 지원하는

**AI Native Customer Experience Platform**을 구축하는 것이 최종 목표이다.
