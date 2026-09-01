//! C-ABI exports for the wasm build. The JS side copies input into a buffer obtained from
//! `tokzip_alloc`, calls `tokzip_compress`/`tokzip_decompress`, and reads the result from the
//! module-owned output buffer (`tokzip_out_ptr`/`tokzip_out_len`), which stays valid until the
//! next call. wasm32-unknown-unknown is single-threaded, so the output buffer is a thread-local.

use std::cell::RefCell;

thread_local! {
    static OUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static IS_BYTES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Allocates `len` bytes (at least one) for the JS side to fill; released by `tokzip_free`.
#[no_mangle]
pub extern "C" fn tokzip_alloc(len: usize) -> *mut u8 {
    let layout = input_layout(len);
    // SAFETY: the layout has a nonzero size.
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    ptr
}

/// # Safety
/// `ptr` must come from `tokzip_alloc(len)` with the same `len` and not have been freed.
#[no_mangle]
pub unsafe extern "C" fn tokzip_free(ptr: *mut u8, len: usize) {
    std::alloc::dealloc(ptr, input_layout(len));
}

fn input_layout(len: usize) -> std::alloc::Layout {
    std::alloc::Layout::from_size_align(len.max(1), 1).expect("input length overflows a layout")
}

/// # Safety
/// `ptr` must come from `tokzip_alloc(len)` (so it is non-null even when `len` is 0) with
/// `ptr..ptr+len` initialized.
#[no_mangle]
pub unsafe extern "C" fn tokzip_compress(ptr: *const u8, len: usize, is_bytes: u32) {
    let input = std::slice::from_raw_parts(ptr, len);
    let frame = tokzip::compress(input, is_bytes != 0);
    OUT.with(|out| *out.borrow_mut() = frame);
}

/// Returns 0 on success (content in the output buffer, type flag via `tokzip_out_is_bytes`),
/// or a nonzero `DecodeError` code; a frame declaring more than `max_len` bytes is rejected.
///
/// # Safety
/// `ptr` must come from `tokzip_alloc(len)` (so it is non-null even when `len` is 0) with
/// `ptr..ptr+len` initialized.
#[no_mangle]
pub unsafe extern "C" fn tokzip_decompress(ptr: *const u8, len: usize, max_len: usize) -> u32 {
    let frame = std::slice::from_raw_parts(ptr, len);
    match tokzip::decompress(frame, max_len) {
        Ok((content, is_bytes)) => {
            OUT.with(|out| *out.borrow_mut() = content);
            IS_BYTES.with(|b| b.set(is_bytes));
            0
        }
        Err(error) => error.code(),
    }
}

#[no_mangle]
pub extern "C" fn tokzip_out_is_bytes() -> u32 {
    IS_BYTES.with(|b| b.get()) as u32
}

#[no_mangle]
pub extern "C" fn tokzip_out_ptr() -> *const u8 {
    OUT.with(|out| out.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn tokzip_out_len() -> usize {
    OUT.with(|out| out.borrow().len())
}
