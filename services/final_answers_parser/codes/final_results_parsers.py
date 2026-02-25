# Accessing the first item directly
final_results = {}
final_input_results = {}

for item in _input.all():
  if 'email_payload_bin' in item.json:
    email_json = item.json.get('email_payload_bin')
    final_input_results["email_payload_bin"] = email_json
  elif 'chunks_bin' in item.json:
    chunks_json = item.json.get('chunks_bin')
    final_input_results["chunks_bin"] = chunks_json

final_results["overall_status"] = final_input_results["email_payload_bin"]["overall_status"]
final_results['answers'] = []
citations_chunks = final_input_results["chunks_bin"]["chunks"]

for result in final_input_results["email_payload_bin"]["results"]:
    single_answer = {}
    single_answer["q_id"] = result["q_id"]
    single_answer["status"] = result["status"]
    single_answer["question"] = result["question"]

    if result["status"] == "ok":
      single_answer["answer"] = result["answer"]
      single_answer['citations'] = []

      for citation in result["citations"]:
          single_citation = {}
          single_citation['citation_id'] = citation
          single_citation['citation_text'] = ""

          for chunk in citations_chunks:
            if chunk['chunk_id'] == citation:
                single_citation['citation_text'] = chunk['content']
                break

          single_answer['citations'].append(single_citation)

      if ('judge' in result) and result['judge'] is not None:
        single_answer['evaluation_judge_scores'] = result['judge']["scores"]
        single_answer['evaluation_judge_verdict'] = result['judge']["verdict"]

    single_answer['user_feedback'] = result['user_feedback']
    final_results['answers'].append(single_answer)

# Your logic here
return final_results