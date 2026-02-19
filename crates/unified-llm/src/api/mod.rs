// api/mod.rs — High-level API module (Layer 4).

pub mod generate;
pub mod generate_object;
pub mod generate_types;
pub mod stream;
pub mod stream_object;
pub mod tool_loop;
pub mod types;

pub use types::*;

// Re-export the key public functions for ergonomic access.
pub use generate::generate;
pub use generate_object::generate_object;
pub use stream::stream;
pub use stream::stream_with_default;
pub use stream::StreamResult;
pub use stream::TextDeltaStream;
pub use stream_object::stream_object;
pub use stream_object::stream_object_with_default;
pub use stream_object::PartialObject;
pub use stream_object::StreamObjectResult;
