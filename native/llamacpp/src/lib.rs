use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};
use rustler::{Env, NifStruct, ResourceArc, Term};
use std::ops::Deref;

pub struct ExLLamaRef(pub LLama);

#[derive(NifStruct)]
#[module = "LLamaCpp.Model"]
pub struct ExLLama {
    pub resource: ResourceArc<ExLLamaRef>,
}

impl ExLLama {
    pub fn new(llama: LLama) -> Self {
        Self {
            resource: ResourceArc::new(ExLLamaRef::new(llama)),
        }
    }
}

impl ExLLamaRef {
    pub fn new(llama: LLama) -> Self {
        Self(llama)
    }
}

impl Deref for ExLLama {
    type Target = LLama;

    fn deref(&self) -> &Self::Target {
        &self.resource.0
    }
}

unsafe impl Send for ExLLamaRef {}
unsafe impl Sync for ExLLamaRef {}

#[derive(NifStruct)]
#[module = "LLamaCpp.ModelOptions"]
struct ExModelOptions {
    pub context_size: i32,
    pub seed: i32,
    pub n_batch: i32,
    pub f16_memory: bool,
    pub m_lock: bool,
    pub m_map: bool,
    pub low_vram: bool,
    pub vocab_only: bool,
    pub embeddings: bool,
    pub n_gpu_layers: i32,
    pub main_gpu: String,
    pub tensor_split: String,
    pub numa: bool,
}

impl From<ExModelOptions> for ModelOptions {
    fn from(a: ExModelOptions) -> Self {
        ModelOptions {
            context_size: a.context_size,
            seed: a.seed,
            n_batch: a.n_batch,
            f16_memory: a.f16_memory,
            m_lock: a.m_lock,
            m_map: a.m_map,
            low_vram: a.low_vram,
            vocab_only: a.vocab_only,
            embeddings: a.embeddings,
            n_gpu_layers: a.n_gpu_layers,
            main_gpu: a.main_gpu,
            tensor_split: a.tensor_split,
            numa: a.numa,
        }
    }
}

#[derive(NifStruct)]
#[module = "LLamaCpp.PredictOptions"]
struct ExPredictOptions {
    pub seed: i32,
    pub threads: i32,
    pub tokens: i32,
    pub top_k: i32,
    pub repeat: i32,
    pub batch: i32,
    pub n_keep: i32,
    pub top_p: f32,
    pub temperature: f32,
    pub penalty: f32,
    pub f16_kv: bool,
    pub debug_mode: bool,
    pub stop_prompts: Vec<String>,
    pub ignore_eos: bool,
    pub tail_free_sampling_z: f32,
    pub typical_p: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub mirostat: i32,
    pub mirostat_eta: f32,
    pub mirostat_tau: f32,
    pub penalize_nl: bool,
    pub logit_bias: String,
    pub path_prompt_cache: String,
    pub m_lock: bool,
    pub m_map: bool,
    pub prompt_cache_all: bool,
    pub prompt_cache_ro: bool,
    pub main_gpu: String,
    pub tensor_split: String,
}
impl From<ExPredictOptions> for PredictOptions {
    fn from(a: ExPredictOptions) -> Self {
        PredictOptions {
            seed: a.seed,
            threads: a.threads,
            tokens: a.tokens,
            top_k: a.top_k,
            repeat: a.repeat,
            batch: a.batch,
            n_keep: a.n_keep,
            top_p: a.top_p,
            temperature: a.temperature,
            penalty: a.penalty,
            f16_kv: a.f16_kv,
            debug_mode: a.debug_mode,
            stop_prompts: a.stop_prompts,
            ignore_eos: a.ignore_eos,
            tail_free_sampling_z: a.tail_free_sampling_z,
            typical_p: a.typical_p,
            frequency_penalty: a.frequency_penalty,
            presence_penalty: a.presence_penalty,
            mirostat: a.mirostat,
            mirostat_eta: a.mirostat_eta,
            mirostat_tau: a.mirostat_tau,
            penalize_nl: a.penalize_nl,
            logit_bias: a.logit_bias,
            path_prompt_cache: a.path_prompt_cache,
            m_lock: a.m_lock,
            m_map: a.m_map,
            prompt_cache_all: a.prompt_cache_all,
            prompt_cache_ro: a.prompt_cache_ro,
            main_gpu: a.main_gpu,
            tensor_split: a.tensor_split,
            token_callback: None,
        }
    }
}

fn on_load(env: Env, _info: Term) -> bool {
    rustler::resource!(ExLLamaRef, env);
    true
}

#[rustler::nif(schedule = "DirtyCpu")]
fn new(path: String, model_options: ExModelOptions) -> Result<ExLLama, ()> {
    let model_options = ModelOptions::from(model_options);
    let llama = LLama::new(path.into(), &model_options).unwrap();
    Ok(ExLLama::new(llama))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn predict(llama: ExLLama, query: String, predict_options: ExPredictOptions) -> String {
    let predict_options = PredictOptions::from(predict_options);
    let result = llama.predict(query.into(), predict_options).unwrap();
    result
}

#[rustler::nif(schedule = "DirtyCpu")]
fn embeddings(llama: ExLLama, query: String, predict_options: ExPredictOptions) -> Vec<f32> {
    let mut predict_options = PredictOptions::from(predict_options);
    let result = llama
        .embeddings(query.into(), &mut predict_options)
        .unwrap();
    result
}

#[rustler::nif(schedule = "DirtyCpu")]
fn free_model(llama: ExLLama) {
    llama.free_model();
}

rustler::init!(
    "Elixir.LLamaCpp",
    [predict, new, embeddings, free_model],
    load = on_load
);
