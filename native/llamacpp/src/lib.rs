use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};
use rustler::{Env, NifStruct, ResourceArc, Term};
use std::ops::Deref;
use std::sync::{Arc, Mutex};

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

fn on_load(env: Env, _info: Term) -> bool {
    rustler::resource!(ExLLamaRef, env);
    true
}

#[rustler::nif(schedule = "DirtyCpu")]
fn new(path: String) -> Result<ExLLama, ()> {
    let model_options = ModelOptions::default();
    let llama = LLama::new(path.into(), &model_options).unwrap();
    Ok(ExLLama::new(llama))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn predict(llama: ExLLama, query: String) -> String {
    let result = Arc::new(Mutex::new(String::new()));

    let predict_options = PredictOptions {
        token_callback: Some(Box::new({
            let result = Arc::clone(&result);
            move |token| {
                let mut result = result.lock().unwrap(); // Lock the mutex
                result.push_str(&token); // Append the token
                true // Ensure that this return type is correct
            }
        })),
        ..PredictOptions::default()
    };
    llama.predict(query.into(), predict_options).unwrap();
    let locked = result.lock().unwrap();
    locked.to_string()
}

rustler::init!("Elixir.LLamaCpp", [predict, new], load = on_load);
