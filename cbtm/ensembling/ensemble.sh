#!/bin/bash

#SBATCH --job-name=cbtm_simple
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/inference/run_logs/%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/inference/run_logs/%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --account=aip-craffel
#SBATCH --time=23:00:00

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU resources: $CUDA_VISIBLE_DEVICES"

# Load modules and activate environment
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11
source /home/ehghaghi/projects/aip-craffel/ehghaghi/c-btm-distillation/uv-x86_64-unknown-linux-gnu/distill_env/bin/activate

# Export cache dirs
export SCRATCH="/home/ehghaghi/scratch/ehghaghi"
export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache

# =============================================================================
# SIMPLE CONFIGURATION
# =============================================================================

# Model paths
VECTORIZER_PATH="$SCRATCH/clusters/allenai/tulu-3-sft-mixture/8/tfidf.pkl"
CLUSTER_CENTERS_PATH="$SCRATCH/clusters/allenai/tulu-3-sft-mixture/8/kmeans.pkl"
EXPERT_MODELS_DIR="$SCRATCH/distillation_results"

# Python script
CBTM_SCRIPT="ensemble.py"

# Test context - MODIFY THIS FOR YOUR EXPERIMENT
TEST_CONTEXT="\n\nJokaisesta asiakkaasta tallennetaan asiakasnumero (yksikäsitteinen), nimi, osoite ja sähköpostiosoite. Yhdellä asiakkaalla voi olla monta vakuutusta ja samalla vakuutuksella voi olla monta haltijaa (esim. pariskunta ottaa yhteisen kotivakuutuksen). Vakuutuksista tallennetaan myös vakuutusnumero (yksikäsitteinen), vakuuksen tyyppi (esim. koti- tai autovakuutus) ja vuosimaksu.\nTietokantaan tallennetaan tieto vakuutuksiin liittyvistä (asiakkaalle lähetetyistä) laskuista. Näistä laskuista tiedetään, mihin vakuutukseen ne liittyvät, viitenumero (yksikäsitteinen), laskun päivämäärä, eräpäivä, summa ja laskun tila (esim. maksamatta, maksettu tai mitätöity). Tietokannassa säilytetään myös tietoa vanhoista laskuista, jotka on jo maksettu.\nLisäksi tietokantaan tallennetaan tieto vahinkotapauksista sekä niihin liittyvistä korvaushakemuksista. Jokainen ilmoitettu vahinkotapaus liittyy täsmälleen yhteen vakuutukseen. Vahinkotapauksista tallennetaan lisäksi vahinkonumero (yksikäsitteinen), vahinkopäivä, vahingon kuvaus ja vahinkoon liittyvät korvaushakemukset. Samaan vahinkotapaukseen voi liittyä useita korvaushakemuksia (esimerkiksi sairaus, josta tulee eri kuluja monta kertaa usean eri vuoden aikana). Samaan vahinkotapaukseen liittyvät korvaushakemukset erotetaan toisistaan antamalla niille oma numero. Tämä numero on kuitenkin yksikäsitteinen vain samaan vahinkotapaukseen liittyvillä hakemuksilla. Korvaushakemuksista tallennetaan myös haettu korvaussumma, hakemuksen päivämäärä ja päätös (korvattava summa)."

# Parameters
TEMPERATURE="0.1"
TOP_K="2"
MAX_TOKENS="1024"

# =============================================================================
# RUN C-BTM INFERENCE
# =============================================================================

echo "🧠 C-BTM SPARSE ENSEMBLE INFERENCE"
echo "===================================="
echo "📝 Context: $TEST_CONTEXT"
echo "🌡️ Temperature: $TEMPERATURE"
echo "🔝 Top-K Experts: $TOP_K"
echo "🎯 Max Tokens: $MAX_TOKENS"
echo ""

python3 "$CBTM_SCRIPT" \
    --vectorizer "$VECTORIZER_PATH" \
    --cluster-centers "$CLUSTER_CENTERS_PATH" \
    --expert-dir "$EXPERT_MODELS_DIR" \
    --context "$TEST_CONTEXT" \
    --temperature "$TEMPERATURE" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    --generate

echo ""
echo "✅ C-BTM inference complete!"
echo "Job finished at $(date)"