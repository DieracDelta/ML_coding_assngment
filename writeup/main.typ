#import "template.typ": *

#show: project.with(
  title: "582 Project",
  authors: (
    (name: "Justin Restivo", email: "justin.restivo@yale.edu"),
  ),
  date: "February 29, 2024",
)


= Problem Selection

I chose to pursue the "Build a recommender system - predict new songs/playlists for listeners" option suggestion from the assignment suggestions. I chose to implement a naive version of Word2Vec, and my target implementation was in Rust. In this project, I had several primary goals:

- Explore the maturity of the machine learning ecosystem in Rust. With the general industry move to memory safe languages, Rust is is appealing both due to both its type system and memory safety.
- Understand word2vec well enough to implement it from scratch (e.g. without relying explicitly on tensorflow).
-


= DataSet

I used the Million Playlist Dataset, which contains one million playlists and 66346428 unique tracks.
