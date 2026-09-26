# Tests

`uv run pytest -m "not integration and not prolog"` runs the unit tests, including pyright, so one command checks both the contracts' signatures and their meaning. Integration tests call MiniLM, Jev, or MedGemma. To check a new adapter, add it to the role's fixture in `test_contracts.py`. Router tests use a fake classifier, splitter and explainer, so they check the router's rules, not the models' judgment.

| Test | Goal | Type |
|---|---|---|
| `test_policy.py` | | |
| parse_amount_with_comma_and_dollar | "$5,000" parses to 5000 and the payee binds | unit |
| unknown_payee_does_not_bind | A payee outside the allowlist does not bind | unit |
| unauthenticated_wire_denied | An unauthenticated user cannot wire | unit |
| delete_denied_for_customer | Only an admin may delete | unit |
| `test_router.py` | | |
| insufficient_funds_suggests_balance_view | A wire above the balance is denied and suggests the balance view | unit |
| no_suggestion_the_policy_would_deny | An unauthenticated caller with no funds is not offered the balance view | unit |
| authorize_then_settle_debits_once | Settlement debits once per transfer id | unit |
| authorize_refused_when_balance_drained | Authorization is refused when funds are gone | unit |
| jev_none_stops_before_policy | A `none` action is denied before policy | unit |
| jev_wire_is_judged_by_policy | A wire chosen by the classifier is executed only through policy | unit |
| minilm_disagreement_does_not_override_jev | The explainer never replaces the classifier's choice | unit |
| explainer_failure_keeps_the_decision | A broken explainer loses its note, never the decision | unit |
| classifier_failure_denies_without_guessing | A classifier that is down means deny, not a guess | unit |
| route_orders_money_movement_before_deletion | Close + wire: the wire goes first, close is listed | unit |
| route_denied_first_request_offers_no_follow_ups | A denied first request lists nothing | unit |
| route_one_request_uses_the_single_jev_rank | One request skips the split | unit |
| route_failed_split_denies_without_guessing | A failed split is denied | unit |
| route_failed_labeling_denies_without_guessing | A failed labeling is denied | unit |
| route_invalid_order_denies_without_guessing | A reasoner order that repeats or drops a request is denied | unit |
| route_none_label_is_ordered_last | A `none` label goes last | unit |
| live_jev_abstains_on_unrelated_sentence | Real Jev returns `none` for "capital of Portugal" | integration |
| `test_workflow.py` | | |
| denied_request_is_refused_and_never_executes | Path received → routing → refused; no money moves | unit |
| unavailable_classifier_is_refused_in_routing | A classifier that is down ends the request in `refused`, from `routing` | unit |
| approved_wire_completes_with_one_debit | Path received → routing → executing → completed; one debit | unit |
| wire_fails_when_funds_vanish_after_routing | The balance drops after routing decided; execution re-checks it and fails, no debit | unit |
| closed_account_blocks_a_later_wire | A deletion closes the account; a later wire fails | unit |
| closing_twice_fails_the_second_time | A closed account cannot be closed again | unit |
| `test_jev.py` | | |
| unsure_action_and_count_become_none | Jev answers under 0.5 become `none` and an unknown count | unit |
| unsure_label_becomes_none_and_sure_label_is_kept | The same floor applies to split-request labels | unit |
| `test_minilm.py` | | |
| disk_cached_embed_does_not_load_model | A cached query vector skips the model | unit |
| disk_cached_actions_do_not_load_model | Cached action vectors skip the model | unit |
| minilm_agrees_with_wire_for_canonical_sentence | Real MiniLM agrees with the wire and shows no gap | integration |
| `test_contracts.py` | | |
| pyright_reports_no_errors | Every adapter and fake matches its role's signatures; runs pyright | unit |
| intent_rank_rejects_values_outside_the_contract (5 cases) | Unknown action, confidence above 1 or NaN, unknown count, text score: all rejected | unit |
| off_contract_answer_counts_as_unavailable | A confident answer outside the catalog makes the classifier unavailable | unit |
| off_contract_label_counts_as_unavailable | The same for the labeler | unit |
| unavailable_classifier_returns_none (no key, call fails) | An unavailable classifier returns `None`, never raises or guesses | unit |
| unavailable_labeler_returns_none (no key, call fails) | The same for the labeler | unit |
| classifier_answers_within_the_catalog | Actions stay in the catalog plus `none`; the count is a valid `RequestCount` | unit |
| labeler_answers_once_per_request_in_order | One label per request, in the order given | unit |
| unreachable_splitter_raises_instead_of_inventing_requests | A splitter that cannot reach its model raises, so the router denies | unit |
| splitter_keeps_written_order_and_uses_its_settings | Requests come back in written order, from the configured model | unit |
| explainer_flags_a_gap_below_its_floor (0.46, 0.44) | MiniLM flags a gap just below 0.45 and not just above | unit |
| `test_policy_engines.py` | | |
| prolog_policy_agrees_with_python_policy | Prolog returns the same verdict, reasons in order, and suggestion as Python on 1,728 cases | integration (prolog) |
| prolog_policy_orders_requests_like_python_policy | Prolog orders all 252 sequences of 2 or 3 labels as Python does, ties included | integration (prolog) |
| `test_medgemma.py` | | |
| parse_split_keeps_one_request_per_nonblank_line | The split parser keeps one request per line | unit |
| medgemma_splits_requests (4 sentences) | MedGemma splits in the user's words | integration |

Every unit test was checked by planting the bug it guards against and confirming that the test fails. For the contract tests, each of eleven planted bugs failed only the tests meant to catch it.
