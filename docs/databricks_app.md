# Databricks Apps Integration

본 문서는 TV VOC AI Platform을 Databricks Apps와 연결하기 위한 App layer 구조를 설명한다.

## 역할 분리

Databricks App은 분석 배치를 직접 수행하지 않는다.

App의 역할:

* sandbox Delta / UC table 직접 조회
* 분류 현황 확인
* `기타` 리뷰 검토
* 기존 topic 재배치 또는 기타 유지 결정을 `review_decision` 테이블에 저장

Pipeline의 역할:

* Rule Profile / Topic Pool 생성
* Topic Classification
* Classification Full 확장
* 수동 fallback 결정 반영

## Folder Structure

```text
prj_TV_voc/
  app.py
  app.yaml
  manifest.yaml
  requirements.txt

  app/
    config.py
    data_access.py

  app/
    config.py
    data_access.py
```

## Data Flow

```text
Databricks App
    ↓
sandbox.z_jungryo_lee.tv_voc_topic_pool
sandbox.z_jungryo_lee.tv_voc_classification_full
    ↓
사람 검토 / topic 선택
    ↓
sandbox.z_jungryo_lee.tv_voc_review_decision
    ↓
manual fallback apply
    ↓
classification_full update
```

## 실행 순서

1. 기존 taxonomy pipeline 실행
2. `classification_full` 생성 확인
3. Databricks App SQL 접속 환경변수 설정
4. Databricks App에서 `app.py` 실행
5. App에서 기타 리뷰 검토 결과를 `review_decision` 테이블에 저장
6. manual fallback pipeline에서 승인 결정을 반영

## 운영 원칙

App은 배치 로직을 직접 수행하지 않고, 운영 테이블 조회와 검토 결정 저장에 집중한다.

Production 기준 데이터는 기존 Delta / UC table과 pipeline output을 기준으로 관리한다.

## Required App Environment

Databricks SQL connector를 사용한다.

필수 환경변수:

* `DATABRICKS_HOST`
* `DATABRICKS_HTTP_PATH`
* `DATABRICKS_TOKEN`

`VOC_APP_MODEL_KEY`, `VOC_APP_TARGET_CATE_1_DEPTH`, `VOC_APP_TARGET_CATE_2_DEPTH`,
`VOC_APP_TARGET_SC_MEASUREMENT`를 지정하면 App 대상 그룹을 환경변수로 바꿀 수 있다.
