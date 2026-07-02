%md
# VOC AI Platform

> AI Native 기반 고객 경험(Customer Experience) 분석 플랫폼

---

# Project Overview

VOC AI Platform은 온라인 VOC(Voice of Customer)를 AI 기반으로 구조화하여 상품기획 의사결정을 지원하기 위한 분석 플랫폼입니다.

기존 VOC 분석은 리뷰 건수와 긍·부정 비율 중심의 통계 제공에 머물렀습니다.

본 프로젝트는 고객 리뷰를 AI가 구조적으로 이해하고, 고객 경험(Customer Experience)을 데이터로 변환하여 상품기획자가 활용 가능한 인사이트를 제공하는 것을 목표로 합니다.

---

# Vision

기존 분석 방식

```
Review

↓

Positive / Negative

↓

Count
```

VOC AI Platform

```
Customer Review

↓

Customer Experience Structuring

↓

Experience Taxonomy

↓

Driver Analysis

↓

Evidence

↓

AI Insight

↓

Planning Support
```

---

# Project Goal

본 프로젝트의 최종 목표는

**"상품기획 담당자가 고객의 목소리를 데이터 기반으로 이해하고 의사결정에 활용할 수 있는 AI Native 분석 플랫폼 구축"** 입니다.

이를 위해 다음 기능을 단계적으로 구축합니다.

* AI 기반 Experience Taxonomy 구축
* AI Topic Classification
* Customer Experience Driver Analysis
* AI Insight Generation
* Planning Support Agent

---

# Design Philosophy

VOC AI Platform은 단순히 LLM으로 리뷰를 분류하는 프로젝트가 아닙니다.

LLM은 분석을 수행하는 하나의 구성요소이며,

분석 Framework는 Data Scientist가 설계합니다.

즉,

Data Scientist가 분석 구조를 설계하고,

AI는 반복 작업을 수행하며,

최종적으로 상품기획자가 활용 가능한 형태의 분석 결과를 제공합니다.

---

# AI Native Workflow

```
Customer Reviews

↓

Taxonomy Design

↓

Topic Classification

↓

Customer Experience Data

↓

Driver Analysis

↓

Evidence Generation

↓

Insight Generation

↓

Planning Support
```

---

# Core Components

## 1. Experience Taxonomy

고객 경험을 구조화하기 위한 계층형 Topic 체계를 구축합니다.

주요 기능

* Rule Profile 생성
* Topic Pool 생성
* Topic Version 관리
* Taxonomy Validation

---

## 2. Topic Classification

고객 리뷰를 Experience Taxonomy 기반으로 자동 분류합니다.

향후 구조

* Rule-based Classification
* Embedding Search
* Hybrid Classification
* Confidence Score
* Representative Evidence

---

## 3. Driver Analysis

Topic 분포를 넘어

고객 만족도에 영향을 주는 핵심 Driver를 정량적으로 분석합니다.

예시

* Correlation
* Regression
* Market Comparison
* Brand Comparison

---

## 4. AI Insight

Driver 분석 결과를

상품기획자가 활용 가능한 형태의

Insight 및 Action Item으로 자동 생성합니다.

---

# Human in the Loop

VOC AI Platform은 사람을 제거하는 시스템이 아닙니다.

AI가 반복 분석을 수행하고,

사람은

* 분석 기준 설계
* Taxonomy 개선
* 예외 검토
* 결과 검증

에 집중합니다.

향후 Domain Expert Review 프로세스를 지원할 수 있도록 설계합니다.

---

# Development Roadmap

## Phase 1

Experience Taxonomy 구축

* Rule Profile
* Topic Pool
* Topic Classification

---

## Phase 2

Hybrid Classification

* Embedding Search
* Confidence Score
* Validation Framework

---

## Phase 3

Driver Analysis

* Correlation
* Regression
* Driver Selection

---

## Phase 4

AI Insight

* Evidence
* Action Item
* Insight Generation

---

## Phase 5

Planning Support Agent

상품기획자가 자연어 질의를 통해

고객 경험

↓

핵심 Driver

↓

시장 비교

↓

추천 Action

까지 확인할 수 있는 AI Agent를 구축합니다.

---

# Repository Structure

```
voc_ai_platform/

README.md

config.yaml

docs/

src/
```

프로젝트가 확장됨에 따라 Taxonomy, Classification, Driver Analysis, Validation 등의 모듈을 단계적으로 추가합니다.

---

# Development Principles

본 프로젝트는 다음 원칙을 따릅니다.

* Explainable AI
* Reproducible Analysis
* Modular Architecture
* AI Native Workflow
* Human-in-the-Loop
* Data-driven Decision Making

---

# Long-term Vision

VOC AI Platform은 리뷰 분류 시스템이 아닙니다.

고객 경험을 AI가 구조화하고,

상품기획자가 데이터 기반으로 의사결정을 수행할 수 있도록 지원하는

AI Native Customer Experience Platform을 구축하는 것이 최종 목표입니다.
