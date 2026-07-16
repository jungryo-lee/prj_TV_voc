# Databricks Apps Integration

본 문서는 TV VOC AI Platform을 Databricks Apps와 연결하기 위한 App layer 구조를 설명한다.

## 역할 분리

Databricks App은 분석 배치를 직접 수행하지 않는다.

App의 역할:

* App용 Parquet snapshot 조회
* 분류 현황 확인
* `기타` 리뷰 검토
* 기존 topic 재배치 또는 기타 유지 결정 저장

Pipeline의 역할:

* Rule Profile / Topic Pool 생성
* Topic Classification
* Classification Full 확장
* App용 Parquet export
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

  app_data/
    exports/
      topic_pool_current/
      classification_summary/
      others_review_candidates/
    inputs/
      manual_review_decisions/
    outputs/
      manual_fallback_result/
    checkpoints/
      app_run_status/

  src/
    app_support/
      app_exporter.py
    pipeline/
      10_export_app_data.ipynb
```

## Data Flow

```text
Delta / UC tables
    ↓
10_export_app_data.ipynb
    ↓
app_data/exports/*.parquet
    ↓
Databricks App
    ↓
app_data/inputs/manual_review_decisions/*.parquet
    ↓
manual fallback import / apply
    ↓
classification_full / review_decision update
```

## 실행 순서

1. 기존 taxonomy pipeline 실행
2. `classification_full` 생성 확인
3. `src/pipeline/10_export_app_data.ipynb` 실행
4. Databricks App에서 `app.py` 실행
5. App에서 기타 리뷰 검토 결과 저장
6. 저장된 Parquet input을 manual fallback pipeline에서 반영

## 운영 원칙

App용 Parquet은 원천 데이터가 아니라 UI 연동용 snapshot이다.

Production 기준 데이터는 기존 Delta / UC table과 pipeline output을 기준으로 관리한다.
