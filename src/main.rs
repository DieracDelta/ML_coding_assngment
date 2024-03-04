use std::{collections::{HashMap, HashSet}};
use burn::nn::loss::Reduction::Mean;
use std::time::Duration;
use std::thread::sleep;

use burn::{backend::Autodiff, config::Config, data::{dataloader::{batcher::Batcher, DataLoaderBuilder}, dataset::{Dataset, InMemDataset}}, module::{Module, Param}, nn::{loss::{CrossEntropyLoss, CrossEntropyLossConfig, MSELoss, Reduction}, Embedding, EmbeddingConfig, Linear, LinearConfig}, optim::{decay::WeightDecayConfig, momentum::MomentumConfig, AdamConfig, Sgd, SgdConfig}, record::{CompactRecorder, NoStdTrainingRecorder}, tensor::{activation::softmax, backend::{AutodiffBackend, Backend}, ops::IntTensorOps, Data, Distribution, Shape, Tensor}, train::{metric::{store::{Aggregate, Direction, Split}, AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, LossMetric}, ClassificationOutput, LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition, TrainOutput, TrainStep, ValidStep}};
use burn::tensor::ElementConversion;
use burn::tensor::{Int, Float};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;

static ARTIFACT_DIR: &str = "/tmp/burn-mylogs";
static DATA_DIR: &str = "simple_data";
// static DATA_DIR: &str = "data/real_data/data/data";
// static VALID_DIR: &str = "data/real_data/valid_data";

const EPOCH_NUM: usize = 5;
const WINDOW_SIZE: usize = 1;
// actual vocab size, but this is targeting smaller dataset
// const VOCAB_SIZE: usize = 2180152;
// used for prototyping on a smaller dataset
const VOCAB_SIZE: usize =   1002;
const EMBEDDING_SIZE: usize = 32;
const LEARNING_RATE: f64 = 0.2;
// const NUM_UNRELATED_SAMPLES: usize = 512;
// const UNRELATED_SAMPLE_SIZE: usize = 256;

// if you have a GPU
// pub type DEVICE = burn::backend::wgpu::Wgpu;
pub type DEVICE = burn::backend::ndarray::NdArray;


pub fn main() {
    // Obtain the current process ID
    // let pid = std::process::id();
    // Print the PID in case of needing to disable OOM killer for this dataset
    // println!("The PID of this process is: {}", pid);
    // sleep(Duration::from_secs(30));


    use burn::backend::ndarray::{NdArrayDevice};
    // if you have a GPU
    // use burn::backend::wgpu::WgpuDevice;
    // let device = WgpuDevice::VirtualGpu(1);
    let device = NdArrayDevice::Cpu;

    let dataset = load_json_playlists(DATA_DIR.to_string());
    // if we want to print stats on the dataset
    // let (vocab, vocab_size) = gen_stats(&dataset);
    // println!("SIZE OF VOCAB IS {:?}", vocab_size);
    // println!("DATASET: {:?}", <InMemDataset<_> as Dataset<_>>::get(&dataframe, 0));
    //
    // println!("num distinct tracks: {:?}", vocab_size);

    let (training, valid, reverse_mapping) = gen_data_items(dataset);

    train::<Autodiff<DEVICE>>(device, training, valid, 1, EPOCH_NUM, 42, reverse_mapping);
}

pub fn one_hot_vec(idx: i32) -> Vec<i32> {
    // can't let this be an array because would overflow stack
    let mut output = vec![0; VOCAB_SIZE];
    output[idx as usize] = 1;
    output
}

/// generate (training data, validation data, mapping uid -> (song, artist))
pub fn gen_data_items(mut data: Vec<PlaylistShortened>) -> (InMemDataset<MyDataItem>, InMemDataset<MyDataItem>, HashMap<usize, (String, String)>){

    let mut dataset_1 : Vec<MyDataItem> = Vec::with_capacity(data.iter().map(|pl| pl.tracks.len()).sum());
    let mut dataset_2 : Vec<MyDataItem> = vec![];
    let mut mapping : HashMap<(String, String), usize>= HashMap::new();
    let mut mapping_reverse : HashMap<usize, (String, String)> = HashMap::new();
    let mut highest_key = 0;
    let mut idx = 0;
    let mut counter = 0;

    while let Some(playlist) = data.pop() {
        println!("APPENDING A PLAYLIST");
        data.shrink_to_fit();

        let tracks = &playlist.tracks;
        for window in tracks.windows(2) {
            counter += 1;
            let input_track : Vec<TrackShortened> = vec![window[0].clone()];
            let outputs : Vec<TrackShortened> = vec![window[1].clone()];

            let track_idx =
            match mapping.entry((input_track[0].artist_name.to_lowercase(), input_track[0].track_name.to_lowercase())) {
                std::collections::hash_map::Entry::Occupied(o) => *o.get(),
                std::collections::hash_map::Entry::Vacant(v) => {
                    mapping_reverse.insert(highest_key, (input_track[0].artist_name.to_lowercase(), input_track[0].track_name.to_lowercase()));
                    v.insert(highest_key);
                    let r_val = highest_key;
                    highest_key += 1;
                    r_val
                }
            };

            let output_idx =
            match mapping.entry((outputs[0].artist_name.to_lowercase(), outputs[0].track_name.to_lowercase())) {
                std::collections::hash_map::Entry::Occupied(o) => *o.get(),
                std::collections::hash_map::Entry::Vacant(v) => {
                    mapping_reverse.insert(highest_key, (outputs[0].artist_name.to_lowercase(), outputs[0].track_name.to_lowercase()));
                    v.insert(highest_key);
                    let r_val = highest_key;
                    highest_key += 1;
                    r_val
                }
            };

            // TODO get random sample size for negative sampling (if you get to it)

            let data_item =
            MyDataItem {
                input: track_idx as i32,
                output: output_idx as i32
            };

            if idx == 32 {
                idx = 0;
                dataset_2.push(data_item);
            } else {
                dataset_1.push(data_item);
            }

            if counter > 1000 {
                let pl1 = InMemDataset::new(dataset_1.clone());
                let pl2 = InMemDataset::new(dataset_2);

                return (pl1, pl2, mapping_reverse);
            }
        }
    }

    let pl1 = InMemDataset::new(dataset_1.clone());
    let pl2 = InMemDataset::new(dataset_2);

    (pl1, pl2, mapping_reverse)

}

#[derive(Clone, Debug)]
pub struct MyDataItem {
    input: i32/*  [i32; VOCAB_SIZE]>, */,
    output: i32,
    // unused for now, since simplifying
    // _unrelated_samples: [[i32; VOCAB_SIZE]; UNRELATED_SAMPLE_SIZE],
}

pub fn gen_stats(dataset: &Vec<PlaylistShortened>) -> (HashSet<(String, String)>, usize) {
    let mut all_songs :HashSet<(String, String)> = HashSet::new();

    for playlist in dataset.iter() {
        for track in playlist.tracks.iter() {
            all_songs.insert((track.artist_name.to_lowercase().clone(), track.track_name.to_lowercase().clone()));
        }
    }
    let len = all_songs.len();
    (all_songs, len)
}

#[derive(Module, Debug)]
pub struct MyModel<B: Backend> {
    embedded: Linear<B>,
    linear: Linear<B>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct File {
    info: Info,
    playlists: Vec<PlayList>
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlaylistShortened {
    tracks: Vec<TrackShortened>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TrackShortened {
    artist_name: String,
    track_name: String,
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
    #[config(default = 5)]
    pub num_epochs: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = "WINDOW_SIZE")]
    pub window: usize,

    #[config(default = 128)]
    pub batch_size: usize,

    // pub optimizer: SgdConfig
    pub optimizer: AdamConfig
}

#[derive(Clone, Debug)]
pub struct MyDataBatch<B: Backend> {
    /// one hot vec
    pub inputs: Tensor<B, 1, Float>,
    /// target label
    pub targets: Tensor<B, 1, Int>,
}

pub struct MyDataBatcher<B: Backend + 'static> {
    device: B::Device
}

impl<B: Backend> MyDataBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            device: device.clone()
        }
    }
}

impl<B: Backend> Batcher<MyDataItem, MyDataBatch<B>> for MyDataBatcher<B> {
    fn batch(&self, items: Vec<MyDataItem>) -> MyDataBatch<B> {
        // if we were to do actual batching
        // let inputs : Vec<Tensor<B, 1, Int>> =
        //     items
        //     .iter()
        //     .map(|item| Data::<i32,1>::from(item.input))
        //     .map(|data| Tensor::<B, 1, Int>::from_data(data.convert(), &self.device)).collect();
        // let outputs : Vec<Tensor<B, 2, Int>> =
        //     items
        //     .iter()
        //     .map(|item| Data::<i32, 2>::from(item.output))
        //     .map(|data| Tensor::<B, 2, Int>::from_data(data.convert(), &self.device))
        //     .collect()
        //     ;

        let shape = vec![VOCAB_SIZE];

        let inputs : Vec<Tensor<B, 1, Float>> =
            items
            .iter()
            .map(|item| Data::<i32,1>::new(one_hot_vec(item.input), <Shape::<1> as From<&Vec<usize>>>::from(&shape)))
            .map(|data| Tensor::<B, 1, Float>::from_data(data.convert(), &self.device)).collect();
        let outputs : Vec<Tensor<B, 1, Int>> =
            items
            .iter()
            .map(|item| Data::<i32, 1>::from([item.output]))
            .map(|data| Tensor::<B, 1, Int>::from_data(data.convert(), &self.device))
            .collect()
            ;


        MyDataBatch {
            inputs : inputs[0].clone(),
            targets: outputs[0].clone()
        }
    }
}

// iterate through all files in the directory `path`
// that end with `.json`
pub fn load_json_playlists(path: String) -> Vec<PlaylistShortened> {
    // let mut i = 0;
    let mut dataset = vec![];
    for entry in std::fs::read_dir(path).unwrap() {
        // println!("loading file: {i:?}");
        // i += 1;
        let entry = entry.unwrap();
        let path = entry.path();
        // check that file ends with .json
        if path.is_file() && path.extension().is_some() && path.extension().unwrap() == "json" {
            let file = std::fs::File::open(path).unwrap();
            let reader = std::io::BufReader::new(file);
            let file: File = serde_json::from_reader(reader).unwrap();
            let mut playlist : Vec<PlaylistShortened>= file.playlists.into_iter().map(|pl| {
                PlaylistShortened {
                    tracks: pl.tracks.into_iter().map(|track| {
                        TrackShortened {
                            artist_name: track.artist_name,
                            track_name: track.track_name
                        }
                    }).collect()
                }
            }).collect();
            dataset.append(&mut playlist);
        }
        break;
    }
    dataset
}

impl<B: AutodiffBackend> TrainStep<MyDataBatch<B>, ClassificationOutput<B>> for MyModel<B> {
    fn step(&self, batch: MyDataBatch<B>) -> TrainOutput<ClassificationOutput<B>> {

        let classification = self.forward_classification(batch);

        // calculate loss
        let loss = classification.loss.backward();

        TrainOutput::new(self, loss, classification)

    }
}

impl<B: Backend> ValidStep<MyDataBatch<B>, ClassificationOutput<B>> for MyModel<B> {
    fn step(&self, batch: MyDataBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch)
    }
}

// function to actually perform training of data, then example song generation
pub fn train<B: AutodiffBackend>(
    device: B::Device,
    train_data: InMemDataset<MyDataItem>,
    valid_data: InMemDataset<MyDataItem>,
    _batch_size: usize,
    num_epochs: usize,
    seed: u64, reverse_mapping: HashMap<usize, (String, String)>,
    ) {
    let config_optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-2)));

    let config = MyTrainingConfig::new(config_optimizer)
        .with_batch_size(1 /* batch_size */)
        .with_num_epochs(num_epochs)
        .with_seed(seed)
        ;

    let batcher_train =
        MyDataBatcher::<B>::new(device.clone())
        ;


    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(1)
        .shuffle(config.seed)
        .num_workers(9)
        .build(train_data);

    let batcher_valid = MyDataBatcher::<B::InnerBackend>::new(device.clone());
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(1)
        .shuffle(config.seed)
        .num_workers(1)
        .build(valid_data);

    B::seed(config.seed);

    let my_model = MyModel::<B>::new(device.clone());

    // random data used to check that the dimensionality matches when prototyping
    // the learner does this on a separate thread which makes
    // building confusing
    // let fake_random_data = Tensor::<B, 1, Int>::zeros(
    //     [VOCAB_SIZE],
    //     &device,
    // );
    //
    // let fake_random_data_float = Tensor::<B, 1, Float>::zeros(
    //     [VOCAB_SIZE],
    //     &device,
    // );
    //
    // let db = MyDataBatch {
    //     inputs: fake_random_data_float.clone(),
    //     targets: fake_random_data.clone(),
    // };
    //
    // let _result = my_model.forward_classification(db);

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(my_model, config.optimizer.init(), LEARNING_RATE)
        ;

    let model_trained = learner.fit(dataloader_train, dataloader_test);
    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    model_trained.clone()
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");

    // example usage
    let mut my_vec = [0f32; VOCAB_SIZE];
    my_vec[30] = 1.0;

    let my_data =
        Data::<f32, 1>::from(my_vec);

    let my_tensor : Tensor<B, 1, Float> = Tensor::<B, 1, Float>::from_data(my_data.convert(), &device);

    let (author_orig, song_orig) = reverse_mapping.get(&30).unwrap();

    // example usage for generating next song in playlist
    let result = model_trained.forward(my_tensor);
    let maxed = result.clone().argmax(1);
    let result_data = result.to_data().value;
    let data = maxed.to_data().value[0];

    let converted_result = <i32 as ElementConversion>::from_elem(data);
    // println!("resulting index is {converted_result:?}");
    // println!("resulting index is {converted_result:?}");
    let (author, song) = reverse_mapping.get(&(converted_result as usize)).unwrap();
    println!("Listening to {song_orig} by {author_orig} recommended {song} by {author}");

}


impl<B: Backend> MyModel<B> {
    pub fn new(device: B::Device) -> Self {
        let embedded_config = LinearConfig::new(VOCAB_SIZE, EMBEDDING_SIZE).with_bias(false);
        let embedded = embedded_config.init(&device);
        let linear_config = LinearConfig::new(EMBEDDING_SIZE, VOCAB_SIZE);
        let linear = linear_config.init(&device);

        Self {
            embedded,
            linear,
        }

    }
    pub fn forward(
        &self,
        input: Tensor<B, 1, Float>
    ) -> Tensor<B, 2> {
        // let [batch_size, height, width] = input.dims();
        let pre_input : Tensor<B, 2, Float> = Tensor::stack(vec![input], 0).detach();
        // println!("PREINPUT {:?}", pre_input.dims());
        let after_embed = self.embedded.forward(pre_input);
        // println!("AFTER EMBED {:?}", after_embed.dims());
        let after_hidden = self.linear.forward(after_embed);
        // println!("AFTER HIDDEN {:?}", after_hidden.dims());
        let after_softmax = softmax::<2, B>(after_hidden, 1);
        // println!("AFTER SOFTMAX {:?}", after_softmax.dims());

        after_softmax
    }

    pub fn forward_classification(&self, item: MyDataBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.inputs);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device());
        let loss_calculation = loss.forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss: loss_calculation,
            output,
            targets,
        }

    }

}
