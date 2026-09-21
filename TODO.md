# Karaoke AI Agent 개선 TODO

## 우선 수정

- [ ] 프롬프트의 `priority_songs` 참조를 실제 `reserved_songs` 구조와 일치시키기
- [ ] `played_song`의 기본값을 `None`으로 지정하기
- [ ] `played_song`과 `reserved_songs`의 곡 번호를 1 이상으로 검증하기
- [ ] `is_playing`과 `played_song`의 상태 일관성 검증하기
- [ ] 프론트엔드 `background_video_theme`과 백엔드 스키마 불일치 해결하기
- [ ] LLM 호출 및 구조화 출력 검증 실패 시 기존 상태를 유지하도록 예외 처리하기
- [ ] `accepted`와 `changed`를 분리해 정상 멱등 명령과 거부 명령을 구분하기

## LangGraph 전환

- [ ] 전체 `KaraokeMachine` 상태 출력 대신 `CommandDecision`과 action 목록 출력하기
- [ ] `interpret_command` 노드 구현하기
- [ ] `validate_actions` 노드 구현하기
- [ ] `apply_actions` reducer 노드 구현하기
- [ ] `reject_command`와 `build_response` 노드 구현하기
- [ ] 조건부 edge로 정상 처리와 거부 경로 분리하기
- [ ] FastAPI 명령·음성 API를 컴파일된 그래프에 연결하기

## 검증

- [ ] 명령별 reducer 단위 테스트 작성하기
- [ ] 복합 명령의 순차 적용 테스트 작성하기
- [ ] 범위 초과와 미지원 명령의 전체 요청 거부 테스트 작성하기
- [ ] LLM 오류 발생 시 기존 상태 유지 테스트 작성하기
- [ ] Qwen3 1.7B 실제 호출 smoke test 작성하기

## RAG 후속 개선

- [x] 프로젝트 사용 안내 문서 작성 및 출처 표시
- [x] 오른쪽 RAG 패널과 검색 자료 전체 원문 열람
- [ ] 실제 제조사 매뉴얼을 사용하려면 출처·사용 권한 확인 후 별도 추가
- [ ] 한국어 정답·거부 질문 평가셋 확장 및 검색 임계값 조정
- [ ] 문서 변경 시 오래된 Chroma 컬렉션 정리 정책 추가

자세한 구현 순서는 [`docs/LANGGRAPH_PLAN.md`](./docs/LANGGRAPH_PLAN.md)를 참고합니다.
