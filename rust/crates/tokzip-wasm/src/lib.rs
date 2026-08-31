//! C-ABI exports for the wasm build. The JS side copies input into a buffer obtained from
//! `tokzip_alloc`, calls `tokzip_compress`/`tokzip_decompress`, and reads the result from the
//! module-owned output buffer (`tokzip_out_ptr`/`tokzip_out_len`), which stays valid until the
//! next call. wasm32-unknown-unknown is single-threaded, so the output buffer is a thread-local.

use std::cell::RefCell;

thread_local! {
    static OUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[no_mangle]
pub extern "C" fn tokzip_alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len.max(1));
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// # Safety
/// `ptr` must come from `tokzip_alloc(len)` and not have been freed.
#[no_mangle]
pub unsafe extern "C" fn tokzip_free(ptr: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(ptr, 0, len.max(1)));
}

/// # Safety
/// `ptr..ptr+len` must be readable.
#[no_mangle]
pub unsafe extern "C" fn tokzip_compress(ptr: *const u8, len: usize, is_bytes: u32) -> usize {
    let input = std::slice::from_raw_parts(ptr, len);
    let frame = tokzip::compress(input, is_bytes != 0);
    let out_len = frame.len();
    OUT.with(|out| *out.borrow_mut() = frame);
    out_len
}

/// Returns 0 on success (content in the output buffer, type flag via `tokzip_out_is_bytes`),
/// or a nonzero `DecodeError` code.
///
/// # Safety
/// `ptr..ptr+len` must be readable.
#[no_mangle]
pub unsafe extern "C" fn tokzip_decompress(ptr: *const u8, len: usize) -> u32 {
    let frame = std::slice::from_raw_parts(ptr, len);
    match tokzip::decompress(frame) {
        Ok((content, is_bytes)) => {
            OUT.with(|out| *out.borrow_mut() = content);
            IS_BYTES.with(|b| b.set(is_bytes));
            0
        }
        Err(error) => error.code(),
    }
}

thread_local! {
    static IS_BYTES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
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
