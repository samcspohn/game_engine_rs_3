//! `#[derive(Export)]` — the reflection derive (ADR-0010 §3).
//!
//! Emits `engine_core::reflect::Export` plus an inherent `TYPE_NAME`. Fields
//! opt in with `#[export]`; everything else is invisible to the inspector,
//! the save walk and the delta.
//!
//! ```ignore
//! #[derive(Export)]
//! struct MeshRenderer {
//!     #[export(get = mesh_id, set = set_mesh)]
//!     mesh_id: MeshId,
//!     speed: f32, // not exported
//! }
//! ```
//!
//! `get = f` calls `self.f()` and it must return the field's own type by
//! value. `set = g` calls `self.g(transform, value)` — the transform is there
//! because a setter can publish GPU state. See `docs/notes/reflection.md`.

use proc_macro::TokenStream;
use proc_macro_crate::{crate_name, FoundCrate};
use proc_macro2::Span;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Ident, Type};

/// Where to aim generated paths: `engine-core` directly if the deriving crate
/// depends on it, otherwise through the `engine` facade — which is all a game
/// has in its manifest.
fn root() -> proc_macro2::TokenStream {
    let found = |name: &str| {
        let ident = |n: &str| Ident::new(&n.replace('-', "_"), Span::call_site());
        match crate_name(name) {
            Ok(FoundCrate::Itself) => Some(ident(name)),
            Ok(FoundCrate::Name(n)) => Some(ident(&n)),
            Err(_) => None,
        }
    };
    match (found("engine-core"), found("engine")) {
        (Some(core), _) => quote!(::#core),
        (None, Some(facade)) => quote!(::#facade::engine_core),
        (None, None) => quote!(::engine_core),
    }
}

/// How one exported field is read and written.
struct Prop {
    name: Ident,
    ty: Type,
    get: Option<Ident>,
    set: Option<Ident>,
}

#[proc_macro_derive(Export, attributes(export))]
pub fn derive_export(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let root = root();
    let name = &input.ident;
    let (impl_g, ty_g, where_c) = input.generics.split_for_impl();

    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            name,
            "#[derive(Export)] needs a struct with named fields",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(
            name,
            "#[derive(Export)] needs a struct with named fields",
        ));
    };

    let props = fields
        .named
        .iter()
        .filter(|f| f.attrs.iter().any(|a| a.path().is_ident("export")))
        .map(parse_prop)
        .collect::<syn::Result<Vec<_>>>()?;

    let infos = props.iter().map(|p| {
        let (lit, ty) = (p.name.to_string(), &p.ty);
        quote! {
            #root::reflect::PropertyInfo {
                name: #lit,
                kind: <#ty as #root::reflect::Exportable>::KIND,
            }
        }
    });

    let gets = props.iter().map(|p| {
        let lit = p.name.to_string();
        let read = match &p.get {
            Some(g) => quote!(self.#g()),
            None => {
                let f = &p.name;
                quote!(self.#f)
            }
        };
        quote!(#lit => ::core::option::Option::Some(
            #root::reflect::Exportable::to_value(&#read)
        ))
    });

    let sets = props.iter().map(|p| {
        let (lit, ty) = (p.name.to_string(), &p.ty);
        let write = match &p.set {
            Some(s) => quote!(self.#s(transform, v)),
            None => {
                let f = &p.name;
                quote!(self.#f = v)
            }
        };
        quote! {
            #lit => match <#ty as #root::reflect::Exportable>::from_value(value) {
                ::core::option::Option::Some(v) => { #write; true }
                ::core::option::Option::None => false,
            }
        }
    });

    let type_name = name.to_string();
    Ok(quote! {
        impl #impl_g #name #ty_g #where_c {
            /// Stable across builds, unlike `TypeId`, so it can name this
            /// type in a scene file (ADR-0010 §3).
            pub const TYPE_NAME: &'static str = #type_name;
        }

        impl #impl_g #root::reflect::Export for #name #ty_g #where_c {
            fn type_name(&self) -> &'static str {
                #type_name
            }

            fn properties(&self) -> &'static [#root::reflect::PropertyInfo] {
                const PROPS: &[#root::reflect::PropertyInfo] = &[#(#infos),*];
                PROPS
            }

            fn get(&self, name: &str) -> ::core::option::Option<#root::reflect::Value> {
                match name {
                    #(#gets,)*
                    _ => ::core::option::Option::None,
                }
            }

            fn set(
                &mut self,
                name: &str,
                value: #root::reflect::Value,
                transform: &#root::transform::Transform,
            ) -> bool {
                let _ = transform;
                match name {
                    #(#sets,)*
                    _ => false,
                }
            }
        }
    })
}

/// Read one `#[export]` / `#[export(get = f, set = g)]`.
fn parse_prop(f: &syn::Field) -> syn::Result<Prop> {
    let attr = f
        .attrs
        .iter()
        .find(|a| a.path().is_ident("export"))
        .expect("filtered to fields carrying #[export]");
    let (mut get, mut set) = (None, None);
    if !matches!(attr.meta, syn::Meta::Path(_)) {
        attr.parse_nested_meta(|m| {
            let slot = match () {
                _ if m.path.is_ident("get") => &mut get,
                _ if m.path.is_ident("set") => &mut set,
                _ => return Err(m.error("expected `get` or `set`")),
            };
            *slot = Some(m.value()?.parse::<Ident>()?);
            Ok(())
        })?;
    }
    // Half a routing is the dangerous shape: a method-backed setter paired
    // with a field read publishes one and skips the other.
    if get.is_some() != set.is_some() {
        return Err(syn::Error::new_spanned(
            attr,
            "`get` and `set` come as a pair — give both, or neither for plain field access",
        ));
    }
    Ok(Prop {
        name: f.ident.clone().expect("named fields only"),
        ty: f.ty.clone(),
        get,
        set,
    })
}
