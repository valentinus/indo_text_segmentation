from typing import List, Optional
from tqdm import tqdm

import pickle

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, PegasusTokenizer, BartForConditionalGeneration, PegasusForConditionalGeneration
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from transformers import pipeline



def boundaries_to_segments(sentences: List[str], boundaries: List[int]) -> Optional[List[int]]:
    segments = []
    last = 0
    for b in boundaries:
        segments.append(" ".join(sentences[last:b]))
        last = b
    segments.append(" ".join(sentences[last:len(sentences)]))
    return segments

def segments_to_text(sentences: List[str], true_boundaries: List[List[str]]) -> Optional[List[int]]:
    segments = []
    last = 0
    for b in true_boundaries:
        segments.append(" ".join(sentences[last:b]))
        last = b
    segments.append(" ".join(sentences[last:len(sentences)]))
    
    final_text = "\n\n".join(segments)
    return final_text

def generate_summary(articles, model, tokenizer, max_length=1024, LOWER=False):
    summaries = []
    for article in articles:
        if LOWER:
            article = article.lower()
        inputs = tokenizer([article], max_length=max_length, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs["input_ids"])
        summaries.append(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    return summaries

if __name__ == '__main__':
    IS_CNNDM = True
    LOWER = False

    if IS_CNNDM:
        model = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased')
        tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')
    else:
        model = PegasusForConditionalGeneration.from_pretrained('Yale-LILY/brio-xsum-cased')
        tokenizer = PegasusTokenizer.from_pretrained('Yale-LILY/brio-xsum-cased')
    max_length = 1024 if IS_CNNDM else 512
    
    
    with open('results/bert/congen_indobert_base.pkl', 'rb') as file:
        # Load the pickled data
        data = pickle.load(file)

    
    count0_1, count0_2, count0_3 = 0, 0, 0
    for dat in tqdm(data):
        if dat.pk_scores <= 0.1 and dat.windiff_scores <= 0.1 and count0_1 < 201:
            if count0_1 == 0:
                count0_1 += 1
                continue
            if len(dat.sentences) % 8 == 0:
                fixed_boundaries = [i for i in range(8, len(dat.sentences)-1, 8)]
            else:
                fixed_boundaries = [i for i in range(8, len(dat.sentences), 8)]
            our_segmented_text = "\n\n".join(generate_summary(boundaries_to_segments(dat.sentences, dat.pred_boundaries), 
                                                  model, tokenizer, max_length=max_length, LOWER=LOWER))
            with open(f"final_text/pred/01_output_{count0_1-1}.txt", "w") as text_file:
                text_file.write(our_segmented_text)
            
            true_segmented_text = "\n\n".join(generate_summary(boundaries_to_segments(dat.sentences, dat.true_boundaries), 
                                                  model, tokenizer, max_length=max_length, LOWER=LOWER))
            with open(f"final_text/true/01_output_{count0_1-1}.txt", "w") as text_file:
                text_file.write(true_segmented_text)
            
            unsegmented_text = generate_summary([" ".join(dat.sentences)], 
                                                  model, tokenizer, max_length=max_length, LOWER=LOWER)[0]
            with open(f"final_text/unsegmented/01_output_{count0_1-1}.txt", "w") as text_file:
                text_file.write(unsegmented_text)
            
            fixed_segmented_text = "\n\n".join(generate_summary(boundaries_to_segments(dat.sentences, fixed_boundaries), 
                                                  model, tokenizer, max_length=max_length, LOWER=LOWER))
            with open(f"final_text/fixed/01_output_{count0_1-1}.txt", "w") as text_file:
                text_file.write(fixed_segmented_text)
            
            final_text = segments_to_text(dat.sentences, dat.true_boundaries)
            with open(f"final_text/original/01_output_{count0_1-1}.txt", "w") as text_file:
                text_file.write(final_text)
            # print(dat.pk_scores, dat.windiff_scores, fixed_boundaries)
            count0_1 += 1
        elif dat.pk_scores < 0.2 and dat.windiff_scores < 0.2 and count0_2 < 200:
            if len(dat.sentences) % 8 == 0:
                fixed_boundaries = [i for i in range(8, len(dat.sentences)-1, 8)]
            else:
                fixed_boundaries = [i for i in range(8, len(dat.sentences), 8)]
            our_segmented_text = "\n\n".join(generate_summary(boundaries_to_segments(dat.sentences, dat.pred_boundaries), 
                                                  model, tokenizer, max_length=max_length, LOWER=LOWER))
            with open(f"final_text/pred/02_output_{count0_2}.txt", "w") as text_file:
                text_file.write(our_segmented_text)
            
            true_segmented_text = "\n\n".join(generate_summary(boundaries_to_segments(dat.sentences, dat.true_boundaries), 
                                                  model, tokenizer, max_length=max_length, LOWER=LOWER))
            with open(f"final_text/true/02_output_{count0_2}.txt", "w") as text_file:
                text_file.write(true_segmented_text)
            
            unsegmented_text = generate_summary([" ".join(dat.sentences)], 
                                                  model, tokenizer, max_length=max_length, LOWER=LOWER)[0]
            with open(f"final_text/unsegmented/02_output_{count0_2}.txt", "w") as text_file:
                text_file.write(unsegmented_text)
            
            fixed_segmented_text = "\n\n".join(generate_summary(boundaries_to_segments(dat.sentences, fixed_boundaries), 
                                                  model, tokenizer, max_length=max_length, LOWER=LOWER))
            with open(f"final_text/fixed/02_output_{count0_2}.txt", "w") as text_file:
                text_file.write(fixed_segmented_text)
            final_text = segments_to_text(dat.sentences, dat.true_boundaries)
            with open(f"final_text/original/02_output_{count0_2}.txt", "w") as text_file:
                text_file.write(final_text)
            # print(dat.pk_scores, dat.windiff_scores, fixed_boundaries)
            count0_2 += 1
        elif dat.pk_scores <= 0.3 and dat.windiff_scores <= 0.3 and count0_3 < 200:
            if len(dat.sentences) % 8 == 0:
                fixed_boundaries = [i for i in range(8, len(dat.sentences)-1, 8)]
            else:
                fixed_boundaries = [i for i in range(8, len(dat.sentences), 8)]
            our_segmented_text = "\n\n".join(generate_summary(boundaries_to_segments(dat.sentences, dat.pred_boundaries), 
                                                  model, tokenizer, max_length=max_length, LOWER=LOWER))
            with open(f"final_text/pred/03_output_{count0_3}.txt", "w") as text_file:
                text_file.write(our_segmented_text)
            
            true_segmented_text = "\n\n".join(generate_summary(boundaries_to_segments(dat.sentences, dat.true_boundaries), 
                                                  model, tokenizer, max_length=max_length, LOWER=LOWER))
            with open(f"final_text/true/03_output_{count0_3}.txt", "w") as text_file:
                text_file.write(true_segmented_text)
            
            unsegmented_text = generate_summary([" ".join(dat.sentences)], 
                                                  model, tokenizer, max_length=max_length, LOWER=LOWER)[0]
            with open(f"final_text/unsegmented/03_output_{count0_3}.txt", "w") as text_file:
                text_file.write(unsegmented_text)
            
            fixed_segmented_text = "\n\n".join(generate_summary(boundaries_to_segments(dat.sentences, fixed_boundaries), 
                                                  model, tokenizer, max_length=max_length, LOWER=LOWER))
            with open(f"final_text/fixed/03_output_{count0_3}.txt", "w") as text_file:
                text_file.write(fixed_segmented_text)
            final_text = segments_to_text(dat.sentences, dat.true_boundaries)
            with open(f"final_text/original/03_output_{count0_3}.txt", "w") as text_file:
                text_file.write(final_text)
            # print(dat.pk_scores, dat.windiff_scores, fixed_boundaries)
            count0_3 += 1