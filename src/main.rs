use std::{collections::{HashMap, HashSet}};

use burn::{backend::Autodiff, config::Config, data::{dataloader::{batcher::Batcher, DataLoaderBuilder}, dataset::{Dataset, InMemDataset}}, module::{Module, Param}, nn::{loss::{CrossEntropyLoss, CrossEntropyLossConfig, Reduction}, Embedding, EmbeddingConfig, Linear, LinearConfig}, optim::{decay::WeightDecayConfig, momentum::MomentumConfig, AdamConfig, Sgd, SgdConfig}, record::{CompactRecorder, NoStdTrainingRecorder}, tensor::{activation::softmax, backend::{AutodiffBackend, Backend}, ops::IntTensorOps, Data, Shape, Tensor}, train::{metric::{store::{Aggregate, Direction, Split}, AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, LossMetric}, ClassificationOutput, LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition, TrainOutput, TrainStep, ValidStep}};
use burn::tensor::ElementConversion;
use burn::tensor::{Int, Float};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;

// TODOS:
// - fix the output sizes because while we kinda need that, it's also eh
// - implement loss correctly
// - fix gradient descent to match textbook
// - check windowing behaviour, it might be wrong..
// - match embedding layer init/config settings to tf code

static ARTIFACT_DIR: &str = "/tmp/burn-example-mnist";
static DATA_DIR: &str = "data/real_data/data";

const WINDOW_SIZE: usize = 1;
// const VOCAB_SIZE: usize = 2180152;
const VOCAB_SIZE: usize =  34232;
const EMBEDDING_SIZE: usize = 256;
const NUM_UNRELATED_SAMPLES: usize = 512;
const UNRELATED_SAMPLE_SIZE: usize = 256;
const LEARNING_RATE: f64 = 0.1;

pub type DEVICE = burn::backend::ndarray::NdArray;


pub fn main() {
    use burn::backend::ndarray::{NdArrayDevice};
    let device = NdArrayDevice::Cpu;

    // let data : [[[i32; 3]; 2]; 2] = [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]];
    // let mapped_data : Vec<Tensor<Autodiff<DEVICE>, 2, Int>> = data.into_iter().map(|item| Data::<i32, 2>::from(item))
    //     .map(|data| Tensor::<Autodiff<DEVICE>, 2, Int>::from_data(data.convert(), &device))
    //     .collect()
    //
    //     ;
    //
    // let stacked_data = Tensor::<Autodiff<DEVICE>, 2, Int>::stack::<3>(mapped_data, 0);

    // panic!("And we're done {:?}", stacked_data);
    let dataset = load_json_playlists(DATA_DIR.to_string());
    let (vocab, vocab_size) = gen_stats(&dataset);
    println!("SIZE OF VOCAB IS {:?}", vocab_size);
    // println!("DATASET: {:?}", <InMemDataset<_> as Dataset<_>>::get(&dataframe, 0));
    //
    println!("num distinct tracks: {:?}", vocab_size);

    let mapping = gen_mapping(vocab);

    let data_items = gen_data_items(&dataset, mapping);
    let validation_data = InMemDataset::<MyDataItem>::new(vec![]);
    panic!("WE NEARLY DONE");
    train::<Autodiff<DEVICE>>(device, data_items, validation_data, 1, 5, 42);



    // let device = Tensor::<f32>::device();
    // do_training::<f32>(device);
}

pub fn one_hot_vec(idx: i32) -> [i32; VOCAB_SIZE] {
    let mut output = [0; VOCAB_SIZE];
    output[idx as usize] = 1;
    output
}

pub fn one_hot_vec_2d(idx: i32) -> Box<[[i32; VOCAB_SIZE]; 1]> {
    let mut output = Box::new([[0; VOCAB_SIZE]; 1]);
    output[0][idx as usize] = 1;
    output
}

/// generate some data items from a hashmap + playlists
pub fn gen_data_items(data: &InMemDataset<PlayList>, mapping: HashMap<(String, String), u32>) -> InMemDataset<MyDataItem>{

    let mut dataset : Vec<MyDataItem> = vec![];

    for playlist in data.iter() {

        let tracks = playlist.tracks;
        for window in tracks.windows(2) {
            // let outputs : Vec<Track> = window[0..WINDOW_SIZE].iter().chain(window[WINDOW_SIZE + 2..].iter()).cloned().collect();

            let input_track : Vec<Track> = vec![window[0].clone()];
            let outputs : Vec<Track> = vec![window[1].clone()];
            // let input_track = &window[WINDOW_SIZE + 1];

            // let input_track = &window[WINDOW_SIZE + 1];
            let track_idx = *mapping.get(&(input_track[0].artist_name.to_lowercase(), input_track[0].track_name.to_lowercase())).unwrap() as i32;


            // TODO get random sample size for negative sampling (if you get to it)

            dataset.push(MyDataItem {
                input: Box::new(one_hot_vec(track_idx)),
                output: outputs.iter().map(|track| *mapping.get(&(track.artist_name.to_lowercase(), track.track_name.to_lowercase())).unwrap() as i32).map(|idx| one_hot_vec_2d(idx)).collect::<Vec<_>>()[0].clone(),
                // _unrelated_samples: [[0; VOCAB_SIZE]; UNRELATED_SAMPLE_SIZE]
                // unrelated_samples: todo!()
            });

            // let track_idx = *mapping.get(&(outputs[0].artist_name.to_lowercase(), outputs[0].track_name.to_lowercase())).unwrap() as i32;

            // dataset.push(MyDataItem {
            //     input: *one_hot_vec(track_idx),
            //     output: input_track.iter().map(|track| *mapping.get(&(track.artist_name.to_lowercase(), track.track_name.to_lowercase())).unwrap() as i32).map(|idx| *one_hot_vec(idx)).collect::<Vec<_>>().try_into().unwrap(),
            //     _unrelated_samples: [[0; VOCAB_SIZE]; UNRELATED_SAMPLE_SIZE]
            //     // unrelated_samples: todo!()
            // });
        }
    }

    InMemDataset::new(dataset)
}

#[derive(Clone, Debug)]
pub struct MyDataItem {
    input: Box<[i32; VOCAB_SIZE]>,
    output: Box<[[i32; VOCAB_SIZE]; 1]>,
    // unused for now, since this is simple
    // _unrelated_samples: [[i32; VOCAB_SIZE]; UNRELATED_SAMPLE_SIZE],
}

// maps from (song_name, author) -> uid
pub fn gen_mapping(dataset: HashSet<(String, String)>) -> HashMap<(String, String), u32>{
    dataset.iter().enumerate().map(|(i, x)| (x.clone(), i as u32)).collect()
}

pub fn gen_stats(dataset: &InMemDataset<PlayList>) -> (HashSet<(String, String)>, usize) {
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
    embedded: Embedding<B>,
    linear: Linear<B>,
    loss: CrossEntropyLoss<B>
    // activation: CrossEntropyLoss<B>
}

impl<B: Backend> MyModel<B> {
    pub fn parameters(){

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
    pub num_epochs: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = "WINDOW_SIZE")]
    pub window: usize,

    #[config(default = 128)]
    pub batch_size: usize,

    pub optimizer: SgdConfig
}

struct PlaylistToItems;

#[derive(Clone, Debug)]
pub struct MyDataBatch<B: Backend> {
    // name is a vector
    pub inputs: Tensor<B, 1, Int>,
    // XX not anymore targets is a vector of vectors
    // target is a vector of what is expected
    pub targets: Tensor<B, 1, Int>,
    // before batch [[onehot vec]; num_unrelated]
    // after batching, [[[onehot vec]; num_unrelated]; batch_size]
    // so dimension is 3
    // pub unrelated_samples: Tensor<B, 3, Int>,
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

        let inputs : Vec<Tensor<B, 1, Int>> =
            items
            .iter()
            .map(|item| Data::<i32,1>::from(*item.input))
            .map(|data| Tensor::<B, 1, Int>::from_data(data.convert(), &self.device)).collect();
        let outputs : Vec<Tensor<B, 2, Int>> =
            items
            .iter()
            .map(|item| Data::<i32, 2>::from(*item.output))
            .map(|data| Tensor::<B, 2, Int>::from_data(data.convert(), &self.device))
            .collect()
            ;

        // let unrelated_samples : Vec<Tensor<B, 2, Int>> =
        //     items
        //     .iter()
        //     .map(|item| Data::<i32, 2>::from(item.unrelated_samples))
        //     .map(|data| Tensor::<B, 2, Int>::from_data(data.convert(), &self.device))
        //     .collect()
        //     ;
        MyDataBatch {
            inputs : Tensor::stack(inputs, 0),
            targets: Tensor::stack(outputs, 0) // TODO do I need to to_device it?
            // inputs: Tensor::stack(inputs, 0).to_device(&self.device),
            // targets: Tensor::stack(outputs, 0).to_device(&self.device),
            // unrelated_samples: Tensor::stack(unrelated_samples, 0).to_device(&self.device),
        }
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

impl<B: AutodiffBackend> TrainStep<MyDataBatch<B>, ClassificationOutput<B>> for MyModel<B> {
    fn step(&self, batch: MyDataBatch<B>) -> TrainOutput<ClassificationOutput<B>> {

        let classification = self.forward_classification(batch);

        TrainOutput::new(self, classification.loss.backward(), classification)


        // compute


        // let input = song_artist_as_vec_list;
        // /// need to cat all the tensors together.
        // let output = Tensor::reshape(targets, todo!());
        // self.embedded.forward();


        // let item = self.forward_classification(item);
        //
        // TrainOutput::new(self, item.loss.backward(), item)
    }
}

// TODO
impl<B: Backend> ValidStep<MyDataBatch<B>, ClassificationOutput<B>> for MyModel<B> {
    fn step(&self, item: MyDataBatch<B>) -> ClassificationOutput<B> {
        todo!()
    }
}

// equivalent to training.rs::run
pub fn train<B: AutodiffBackend>(
    device: B::Device,
    train_data: InMemDataset<MyDataItem>,
    valid_data: InMemDataset<MyDataItem>,
    batch_size: usize,
    num_epochs: usize,
    seed: u64) {
    // copied from mnist example. IDK if it's any good.
    // let config_optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
    let config_optimizer = SgdConfig::new();

    // TODO set params properly
    let config = MyTrainingConfig::new(config_optimizer)
        .with_batch_size(batch_size)
        .with_num_epochs(num_epochs)
        .with_seed(seed)
        ;

    let batcher_train =
        MyDataBatcher::<B>::new(device.clone())
        ;

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(1)
        .shuffle(config.seed)
        .num_workers(1)
        .build(train_data);

    let batcher_valid = MyDataBatcher::<B::InnerBackend>::new(device.clone());
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(1)
        .shuffle(config.seed)
        .num_workers(1)
        .build(valid_data);


    B::seed(config.seed);

    let my_model = MyModel::<B>::new(device.clone());

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
        // stop early if no improvement
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        // TODO do I need an optimizer
        .build(my_model, config.optimizer.init(), LEARNING_RATE)
        ;

    let model_trained = learner.fit(dataloader_train, dataloader_test);
    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    model_trained
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");

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
        let embedded_config = EmbeddingConfig::new(VOCAB_SIZE, EMBEDDING_SIZE);
        let embedded = embedded_config.init(&device);
        let linear_config = LinearConfig::new(EMBEDDING_SIZE, VOCAB_SIZE);
        let linear = linear_config.init(&device);
        let cross_entropy_loss_config = CrossEntropyLossConfig::new();
        let cross_entropy_loss = cross_entropy_loss_config.init(&device);

        Self {
            embedded,
            linear,
            loss: cross_entropy_loss
        }

    }

    pub fn forward(
        &self,
        input: Tensor<B, 1, Int>
        // logits: Tensor<B, D>,
        // targets: Tensor<B, D>,
        // reduction: Reduction
    ) -> Tensor<B, 2> {
        // let [batch_size, height, width] = input.dims();
        // TODO might be wrong way, eg width*batch_size instead
        // let input_resized = input.reshape([batch_size * height, width]);
        let after_embed = self.embedded.forward(Tensor::stack(vec![input], 0));
        let after_hidden = self.linear.forward(after_embed);
        // TODO find out what dimension this is
        let after_softmax = softmax::<2, B>(after_hidden.reshape([1, EMBEDDING_SIZE]), 0);


        after_softmax
    }

    pub fn forward_classification(&self, item: MyDataBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.inputs);
        let loss_calculation = self.loss.forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss: loss_calculation,
            output,
            targets,
        }

    }

}
