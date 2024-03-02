#import "template.typ": *

#show: project.with(
  title: "582 Coding Project",
  authors: (
    (name: "Justin Restivo", email: "justin.restivo@yale.edu"),
  ),
  date: "February 29, 2024",
)


= Problem Selection

I chose to pursue the "Build a recommender system - predict new songs/playlists for listeners" option suggestion from the assignment suggestions. I chose to implement a naive version of Word2Vec, and my target implementation was in Rust, using the [Burn crate](https://burn.dev/). In this project, I had several primary goals:

- Explore the maturity of the machine learning ecosystem in Rust.
- Understand Word2Vec well enough to implement it from scratch (e.g. without relying explicitly on tensorflow).

= Building the project

To build the executable to run the project on the dataset, first install nix using the determinate systems nix installer (or, if nix is already installed, enable flakes):

```bash
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```

Then fill the data directory with data TODO explain how to do this

Finally in the root directory of the project, run the executable:

```bash
nix develop -c cargo run --release
```

If there are issues with this step, please reach out to me.

= Design Decisions

== Dataset

I first looked through the Million Playlist Dataset (MPD) from spotify. This was appealing due to the sheer amount of data. MPD contained 1000000 playlists and 66346428 unique tracks, as well as an evaluation metric for "goodness" of the predictions (TODO verify this). With the target of playlist generation, I wanted to find a reasonable algorithm to fully utilize this dataset. Word2vec is a quite natural choice here, since with the skip-gram model, we are able to predict associated songs. Intuitively this seemed like a good direction.

== Rust as an implementation language

I spent ~5 years in industry prior to my PHD, and in that time developed a belief that memory safe, strongly/statically typed languages like Rust are the future for low level systems. In fact, the government agrees TODO ref. Some rationale for this high level design decision:

- Expressive type system
- Speed without sacrificing memory safety
- Fantastic tooling
- Static and strongly typed which makes runtime errors are simple to debug, and pushes most runtime errors back to compile-time errors that can be easily solved.

As a result, I wanted to use this project as an opportunity to explore machine learning with Rust. Furthermore, since the Rust ecosystem is not as mature as Python, this seemed like an especially good opportunity to "implement state-of-the-art algorithms" as the instructions requested from scratch, thereby forcing me to learn more than I would have otherwise by relying on library functions.

Finally, prototyping seemed easier to me because I'm able to avoid runtime typing errors, which frequently happened to me when I used to write python.

= Word2Vec

Word2Vec

// talk about literature, link blog posts, discuss negative sampling and NCE loss
//

= Burn

== High level abstractions

== Loading the dataset

// had to write loader
//

== Constructing the model

== Training the model

= Results


