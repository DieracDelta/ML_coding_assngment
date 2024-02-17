use std::{marker::PhantomData};

use burn::{config::Config, data::{dataloader::batcher::Batcher, dataset::{Dataset, InMemDataset}}, module::Module, nn::{Embedding, EmbeddingConfig}, optim::{decay::WeightDecayConfig, AdamConfig}, record::CompactRecorder, tensor::{backend::{AutodiffBackend, Backend}, Tensor}, train::{metric::{store::{Aggregate, Direction, Split}, AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, LossMetric}, LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition}};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;

static ARTIFACT_DIR: &str = "/tmp/burn-example-mnist";
static DATA_DIR: &str = "data/real_data/data";

pub fn main() {
    let dataframe = load_json_playlists(DATA_DIR.to_string());
    println!("DATASET: {:?}", <InMemDataset<_> as Dataset<_>>::get(&dataframe, 0));
    // let device = Tensor::<f32>::device();
    // do_training::<f32>(device);
}

#[derive(Module, Debug)]
pub struct MyModel<B: Backend> {
    embedded: Embedding<B>
}

impl<B: Backend> MyModel<B> {
    pub fn parmeters(){

    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct File {
    info: Info,
    playlists: Vec<PlayList>
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Info {
    generated_on: String,
    slice: String,
    version: String
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlayList {
    name: String,
    collaborative: String,
    pid: isize,
    modified_at: isize,
    num_tracks: isize,
    num_albums: isize,
    num_followers: isize,
    tracks: Vec<Track>,
    num_edits: isize,
    duration_ms: isize,
    num_artists: isize
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Track {
    pos: isize,
    artist_name: String,
    track_uri: String,
    artist_uri: String,
    track_name: String,
    album_uri: String,
    duration_ms: isize,
    album_name: String
}

#[derive(Config)]
pub struct MyTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: isize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig
}

pub struct MyDataItem {
}

pub struct MyDataBatch<B: Backend> {
    _pd: PhantomData<B>
}

pub struct MyDataBatcher<B: Backend> {
    device: B::Device
}

impl<B: Backend> MyDataBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            device
        }
    }
}

impl<B: Backend> Batcher<MyDataItem, MyDataBatch<B>> for MyDataBatcher<B> {
    fn batch(&self, items: Vec<MyDataItem>) -> MyDataBatch<B> {
        todo!()
    }
}
// iterate through all files in the directory `path`
// that end with `.json`
pub fn load_json_playlists(path: String) -> InMemDataset<PlayList> {
    let mut dataset = vec![];
    for entry in std::fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        // check that file ends with .json
        if path.is_file() && path.extension().is_some() && path.extension().unwrap() == "json" {
            let file = std::fs::File::open(path).unwrap();
            let reader = std::io::BufReader::new(file);
            let file: File = serde_json::from_reader(reader).unwrap();
            let mut playlist = file.playlists;
            dataset.append(&mut playlist);
        }
    }
    InMemDataset::new(dataset)
}

// equivalent to training.rs::run
pub fn do_training<B: AutodiffBackend>(device: B::Device) {
    // copied from mnsit example. IDK if it's any good.
    let config_optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));

    // TODO set params properly
    let config = MyTrainingConfig::new(config_optimizer);





    B::seed(config.seed);

    let batcher_train = MyDataBatcher::<B>::new(device.clone());
    let batcher_validate = MyDataBatcher::<B>::new(device.clone());

    todo!()
    // let my_model = MyModel::new(device.clone());

    // let learner = LearnerBuilder::<B>::new(ARTIFACT_DIR)
    //     .metric_train_numeric(AccuracyMetric::new())
    //     .metric_valid_numeric(AccuracyMetric::new())
    //     .metric_train_numeric(CpuUse::new())
    //     .metric_valid_numeric(CpuUse::new())
    //     .metric_train_numeric(CpuMemory::new())
    //     .metric_valid_numeric(CpuMemory::new())
    //     .metric_train_numeric(CpuTemperature::new())
    //     .metric_valid_numeric(CpuTemperature::new())
    //     .metric_train_numeric(LossMetric::new())
    //     .metric_valid_numeric(LossMetric::new())
    //     .with_file_checkpointer(CompactRecorder::new())
    //     .devices(vec![device.clone()])
    //     .num_epochs(config.num_epochs)
    //     // TODO not sure about this?
    //     .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
    //         Aggregate::Mean,
    //         Direction::Lowest,
    //         Split::Valid,
    //         StoppingCondition::NoImprovementSince { n_epochs: 1 },
    //     ))
    //     // TODO do I need an optimizer
    //     .build(my_model, config.optimizer.init(), /* lr_scheduler. no idea what this does*/ 1e-4)
    //     ;






    //
    // let model = MyModel::new(device);
    // burn::nn::Embedding::new(10, 10, &device);
    //
    // // let mut optimizer = burn::nn::Adam::new(model.parameters(), 0.001);
    // // let mut train_loader = MyDataBatcher {
    // //     _pd: PhantomData
    // // };
    // // for epoch in 0..config.num_epochs {
    // //     for batch in train_loader {
    // //         let input = Tensor::randn([config.batch_size, 28 * 28], &device);
    // //         let target = Tensor::randn([config.batch_size, 10], &device);
    // //         let output = model.forward(input);
    // //         let loss = burn::nn::cross_entropy_loss(&output, &target);
    // //         optimizer.backward_step(&loss);
    // //     }
    // // }
    //
    //
}


impl<B: Backend> MyModel<B> {
    pub fn new(device: B::Device) -> Self {
        let embedded_config = EmbeddingConfig::new(todo!(), todo!());
        let embedded = embedded_config.init(&device);

        Self {
            embedded
        }

    }

    pub fn forward<const N: usize>(&self, input: Tensor<B, N>) -> Tensor<B, N> {
        todo!()
    }

}

