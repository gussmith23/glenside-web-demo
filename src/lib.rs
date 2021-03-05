#![recursion_limit = "256"]

use wasm_bindgen::prelude::*;
use yew::{html, App, Component, ComponentLink, Html, ShouldRender};

pub struct Model {
    link: ComponentLink<Self>,
    value: usize,
}

impl Component for Model {
    type Message = ();
    type Properties = ();

    fn create(_props: Self::Properties, link: ComponentLink<Self>) -> Self {
        Self { link, value: 0 }
    }

    fn update(&mut self, _msg: Self::Message) -> ShouldRender {
        false
    }

    fn change(&mut self, _props: Self::Properties) -> ShouldRender {
        false
    }

    fn view(&self) -> Html {
        html! {
            <div>
                <a href={"https://github.com/gussmith23/glenside"}>{"Github repo"}</a>
                <nav class="menu">
                    <button onclick=self.link.callback(|_| ())>
                        { "Increment" }
                    </button>
                    <button onclick=self.link.callback(|_| ())>
                        { "Decrement" }
                    </button>
                </nav>
                <p>
                    <b>{ "Current value: " }</b>
                    { self.value }
                </p>
            </div>
        }
    }
}

#[wasm_bindgen(start)]
pub fn run_app() {
    App::<Model>::new().mount_to_body();
}
