#![recursion_limit = "1024"]

use glenside::language::interpreter::Environment;
use lazy_static::lazy_static;
use monaco::{
    api::CodeEditorOptions,
    sys::editor::BuiltinTheme,
    yew::{CodeEditor, CodeEditorLink},
};
use ndarray::{ArrayD, Dimension, IxDyn};
use rand::{
    distributions::{Distribution, Uniform},
    rngs::OsRng,
};
use std::collections::HashMap;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use yew::{html, html_nested, Component, ComponentLink, Html, InputData, Properties, ShouldRender};

fn get_options() -> CodeEditorOptions {
    CodeEditorOptions::default()
        .with_new_dimension(500, 500)
        .with_builtin_theme(BuiltinTheme::VsDark)
}

struct Example<'a> {
    name: &'a str,
    description: &'a str,
    glenside_source: &'a str,
    environment: Environment<'a, f64>,
}

lazy_static! {
    static ref CONV: Example<'static> = Example {
        name: "2D Convolution",
        description: "",
        glenside_source: r#"
(access-transpose
 (compute dot-product
  (access-cartesian-product
   (access (access-tensor weights) 1)
   (access
    (access-squeeze
     (access-squeeze
      (access-windows
       (access
        (access-pad
         (access-pad
          (access-tensor activations)
          zero-padding 2 1 1)
         zero-padding 3 1 1)
        4)
       (shape 1 3 3 3)
       (shape 1 1 1 1))
      4)
     1)
    3)))
 (list 1 0 2 3))
"#,
        environment: {
            let mut env = HashMap::new();
            env.insert(
                "activations",
                ArrayD::from_shape_fn(IxDyn(&[1, 3, 32, 32]), |_| {
                    Uniform::new(-2.0, 2.0).sample(&mut OsRng::new().unwrap())
                }),
            );
            env.insert(
                "weights",
                ArrayD::from_shape_fn(IxDyn(&[8, 3, 3, 3]), |_| {
                    Uniform::new(-2.0, 2.0).sample(&mut OsRng::new().unwrap())
                }),
            );
            env
        },
    };
}

lazy_static! {
    static ref EXAMPLES: Vec<&'static Example<'static>> = vec![&CONV];
}

enum Message {
    NewInput,
    EnvironmentValueUpdated(String, ArrayD<f64>),
    AddNewEnvironmentInput,
    ExampleSelected(Option<usize>),
}

struct App {
    options: Rc<CodeEditorOptions>,
    link: ComponentLink<Self>,
    code_editor_link: CodeEditorLink,
    result_text: String,
    environment: Environment<'static, f64>,
    num_environment_inputs: usize,
    /// Stores whatever the user has typed into the editor. Used when switching
    /// back and forth between examples, so we can save/restore whatever the
    /// user has typed.
    user_editor_state: String,
    /// Stores whatever environment the user has entered. Used when switching
    /// back and forth between examples, so that we can save/restore the user's
    /// environment.
    user_environment_state: Environment<'static, f64>,
    pre_set_environment: Option<Environment<'static, f64>>,
    example_selected: Option<usize>,
}
impl Component for App {
    type Message = Message;
    type Properties = ();

    fn create(_props: Self::Properties, link: ComponentLink<Self>) -> Self {
        Self {
            options: Rc::new(get_options()),
            link: link,
            code_editor_link: CodeEditorLink::default(),
            result_text: String::default(),
            environment: Environment::default(),
            num_environment_inputs: 0,
            user_editor_state: String::default(),
            user_environment_state: Environment::default(),
            pre_set_environment: None,
            example_selected: None,
        }
    }

    fn update(&mut self, msg: Self::Message) -> ShouldRender {
        match msg {
            Message::AddNewEnvironmentInput => {
                self.num_environment_inputs += 1;
                true
            }
            Message::EnvironmentValueUpdated(name, value) => {
                let name = Box::leak(name.into_boxed_str());
                self.user_environment_state.insert(name, value.clone());
                self.environment.insert(name, value);
                false
            }
            Message::NewInput => {
                let text_input = self
                    .code_editor_link
                    .with_editor(|editor| editor.get_model().unwrap().get_value())
                    .unwrap();

                let result = glenside::language::interpreter::interpret_from_str::<f64>(
                    &text_input,
                    &self.environment,
                );

                let text_output = match result {
                    glenside::language::interpreter::Value::Tensor(t) => t.to_string(),
                    glenside::language::interpreter::Value::Access(a) => {
                        format!(
                            "shape: (({a}), ({b}))\n\
                             value:\n\
                             {tensor}",
                            a = a.tensor.shape()[..a.access_axis]
                                .iter()
                                .map(|i| i.to_string())
                                .collect::<Vec<_>>()
                                .join(", "),
                            b = a.tensor.shape()[a.access_axis..]
                                .iter()
                                .map(|i| i.to_string())
                                .collect::<Vec<_>>()
                                .join(", "),
                            tensor = a.tensor.to_string()
                        )
                    }
                    glenside::language::interpreter::Value::Usize(_) => todo!(),
                    glenside::language::interpreter::Value::Shape(_) => todo!(),
                    glenside::language::interpreter::Value::ComputeType(_) => todo!(),
                    glenside::language::interpreter::Value::PadType(_) => todo!(),
                    glenside::language::interpreter::Value::AccessShape(shape, access_axis) => {
                        format!(
                            "(({a}), ({b}))",
                            a = shape.slice()[..access_axis]
                                .iter()
                                .map(|i| i.to_string())
                                .collect::<Vec<_>>()
                                .join(", "),
                            b = shape.slice()[access_axis..]
                                .iter()
                                .map(|i| i.to_string())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    }
                    glenside::language::interpreter::Value::List(_) => todo!(),
                };

                self.result_text = text_output;

                true
            }
            Message::ExampleSelected(None) => {
                self.example_selected = None;

                // Restore previous input
                self.options = Rc::new(get_options().with_value(self.user_editor_state.clone()));

                // Restore previous environment
                self.environment = self.user_environment_state.clone();

                self.pre_set_environment = None;

                true
            }
            Message::ExampleSelected(Some(i)) => {
                self.example_selected = Some(i);

                // Save current input
                self.user_editor_state = self
                    .code_editor_link
                    .with_editor(|editor| editor.get_model().unwrap().get_value())
                    .unwrap();

                // Initialize with EXAMPLE[i] Glenside source code
                self.options =
                    Rc::new(get_options().with_value(EXAMPLES[i].glenside_source.to_string()));

                // Take the environment from EXAMPLE[i]
                self.environment = EXAMPLES[i].environment.clone();

                // TODO(@gussmith23) This is super clunky, so much duplicated state.
                self.pre_set_environment = Some(EXAMPLES[i].environment.clone());

                true
            }
        }
    }

    fn change(&mut self, _props: Self::Properties) -> ShouldRender {
        unreachable!()
    }

    fn view(&self) -> Html {
        html! {
            <div>
                <ExampleChooser example_chosen_callback=self.link.callback(|i| Message::ExampleSelected(i)) />
                <EnvironmentInputs
                    value_updated_callback=self.link.callback(|(name, value)| {
                        Message::EnvironmentValueUpdated(name, value)
                    })
                    pre_set_environment={self.example_selected.map(|i| EXAMPLES[i].environment.clone())} />
                <CodeEditor
                    link=&self.code_editor_link
                    options=Rc::clone(&self.options)
                    />
                <input type={"button"} value={"run"} onclick=self.link.callback(|_| Message::NewInput) />
                <br/>
                <textarea readonly={true}>{self.result_text.clone()}</textarea>
            </div>
        }
    }
}

#[derive(Properties, Clone)]
struct ExampleChooserProperties {
    example_chosen_callback: yew::Callback<Option<usize>>,
}
struct ExampleChooser {
    link: ComponentLink<Self>,
    properties: ExampleChooserProperties,
    /// The index into [`EXAMPLES`] of the currently selected example.
    selected_example_index: Option<usize>,
}

enum ExampleChooserMessage {
    Chosen(Option<usize>),
}

impl Component for ExampleChooser {
    type Message = ExampleChooserMessage;
    type Properties = ExampleChooserProperties;

    fn create(props: Self::Properties, link: ComponentLink<Self>) -> Self {
        Self {
            link,
            properties: props,
            selected_example_index: None,
        }
    }

    fn update(&mut self, msg: Self::Message) -> ShouldRender {
        match msg {
            ExampleChooserMessage::Chosen(i) => {
                // Did the selection change?
                let changed = i != self.selected_example_index;

                self.selected_example_index = i;
                self.properties.example_chosen_callback.emit(i);

                changed
            }
        }
    }

    fn change(&mut self, props: Self::Properties) -> ShouldRender {
        self.properties = props;
        true
    }

    fn view(&self) -> Html {
        html! {
            <div>
            {"Choose an example:"}

            <input type={"radio"}
                id={"no-example"}
                name={"example-chooser"}
                checked={self.selected_example_index.is_none()}
                oninput=self.link.callback(move |_| ExampleChooserMessage::Chosen(None))
            />
            <label for={"no-example"}>{"none"}</label>

            {
                for EXAMPLES.iter().enumerate().map(|(i, example)| {
                    html_nested ! {
                        <>
                        <input type={"radio"}
                            id={format!("example-radio-button-{}", i)}
                            name={"example-chooser"}
                            checked={self.selected_example_index==Some(i)}
                            oninput=self.link.callback(move |_| ExampleChooserMessage::Chosen(Some(i)))
                            />
                        <label
                            for={format!("example-radio-button-{}", i)}>
                            {example.name}
                        </label>
                        </>
                    }
                })
            }


            </div>
        }
    }
}

struct EnvironmentInputs {
    props: EnvironmentInputsProps,
    link: ComponentLink<Self>,
    num_environment_inputs: usize,
}

#[derive(Properties, Clone)]
struct EnvironmentInputsProps {
    /// The callback to the parent, which should be called when any of the
    /// environment inputs change. This is the *only* way in which this
    /// component communicates anything about the environment. Messages about
    /// the environment are in the form of (name, tensor) pairs, saying that the
    /// tensor with `name` now has value `tensor`. Thus, there is currently no
    /// way to remove tensors from the environment.
    value_updated_callback: yew::Callback<(String, ArrayD<f64>)>,
    /// A pre-set environment. Currently, the component is just given this for
    /// display purposes. That is, [`value_updated_callback`] is not called for
    /// the tensors in the pre-set environment. The creator of this component is
    /// assumed to put these tensors in their environment manually. In the
    /// future, we could make it so that [`value_updated_callback`] is called
    /// right away for each of the tensors in the pre-set environment.
    #[prop_or_default]
    pre_set_environment: Option<Environment<'static, f64>>,
}

enum EnvironmentInputsMessage {
    Add,
}

impl Component for EnvironmentInputs {
    type Message = EnvironmentInputsMessage;
    type Properties = EnvironmentInputsProps;

    fn create(props: Self::Properties, link: ComponentLink<Self>) -> Self {
        Self {
            props,
            link,
            num_environment_inputs: 0,
        }
    }

    fn update(&mut self, msg: EnvironmentInputsMessage) -> ShouldRender {
        match msg {
            EnvironmentInputsMessage::Add => {
                self.num_environment_inputs += 1;
                true
            }
        }
    }

    fn change(&mut self, props: Self::Properties) -> ShouldRender {
        self.props = props;
        true
    }

    fn view(&self) -> Html {
        html! {
            <div>
                // Button to instantiate inputs
                <input type={"button"} value={"+"} onclick=self.link.callback(|_| EnvironmentInputsMessage::Add) />

                // Pre-set inputs
                {
                    for self.props.pre_set_environment.as_ref().map(|e| e.iter().map(|(name, value)| {
                        html_nested!{
                            <PreSetInput
                                name=name.clone()
                                value=value.clone() />
                        }
                    }).collect::<Vec<_>>()).unwrap_or_default()
                }
                {
                    for (0..self.num_environment_inputs).map(|i| {
                        html_nested!{
                            <GeneratedTensorEnvironmentInput
                                id={i}
                                value_updated_callback=self.props.value_updated_callback.clone() />
                        }
                    })
                }
            </div>
        }
    }
}

struct PreSetInput(PreSetInputProperties);

#[derive(Properties, Clone)]
struct PreSetInputProperties {
    name: String,
    value: ArrayD<f64>,
}

impl Component for PreSetInput {
    type Message = ();
    type Properties = PreSetInputProperties;

    fn create(props: Self::Properties, _link: ComponentLink<Self>) -> Self {
        Self(props)
    }

    fn update(&mut self, _msg: Self::Message) -> ShouldRender {
        unreachable!()
    }

    fn change(&mut self, props: Self::Properties) -> ShouldRender {
        self.0 = props;
        true
    }

    fn view(&self) -> Html {
        html! {
            <div>
                <label for={"name"}>{"Name"}</label>
                <input name={"name"} type={"text"} value={ &self.0.name }
                    disabled={true} />

                // Shape text box
                <label for={"shape"}>{"Shape"}</label>
                <input name={"shape"} type={"text"}
                  value={ format!("({})", self.0.value.shape().iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(",")) }
                  disabled={true}/>
            </div>
        }
    }
}

#[derive(Properties, Clone)]
struct EnvironmentInputProps {
    value_updated_callback: yew::Callback<(String, ArrayD<f64>)>,
    /// Unique id identifying this input in a list of inputs. Currently only
    /// used so that we can make the names of the radio button groups unique.
    id: usize,
}

struct GeneratedTensorEnvironmentInput {
    properties: EnvironmentInputProps,
    link: ComponentLink<Self>,
    name: String,
    shape_string: String,
    value_generation_strategy: Option<ValueGenerationStrategy>,
}

impl GeneratedTensorEnvironmentInput {
    fn get_value(&self) -> Option<(String, ArrayD<f64>)> {
        // First and last characters should be parens.
        if self.shape_string.is_empty()
            || self.shape_string.chars().nth(0).unwrap() != '('
            || self
                .shape_string
                .chars()
                .nth(self.shape_string.len() - 1)
                .unwrap()
                != ')'
        {
            return None;
        }

        let parens_trimmed = &self.shape_string[1..self.shape_string.len() - 1];

        let parse_results = if parens_trimmed.is_empty() {
            vec![]
        } else {
            parens_trimmed
                .split(",")
                .map(|s| s.parse::<usize>())
                .collect::<Vec<_>>()
        };

        if !parse_results.is_empty() && parse_results.iter().any(|r| r.is_err()) {
            return None;
        }

        let shape = parse_results
            .iter()
            .map(|r| *r.as_ref().unwrap())
            .collect::<Vec<_>>();

        match self.value_generation_strategy {
            Some(ValueGenerationStrategy::Zeros) => {
                Some((self.name.clone(), ndarray::ArrayD::zeros(shape)))
            }
            Some(ValueGenerationStrategy::Ones) => {
                Some((self.name.clone(), ndarray::ArrayD::ones(shape)))
            }
            Some(ValueGenerationStrategy::Random) => Some((
                self.name.clone(),
                ndarray::ArrayD::from_shape_fn(shape, |_| {
                    Uniform::new(-2.0, 2.0).sample(&mut OsRng::new().unwrap())
                }),
            )),
            None => None,
        }
    }
}
enum ValueGenerationStrategy {
    Random,
    Zeros,
    Ones,
}

enum GeneratedTensorEnvironmentInputMessage {
    UpdateName(String),
    UpdateShapeString(String),
    UpdateValueGenerationStrategy(ValueGenerationStrategy),
}

impl Component for GeneratedTensorEnvironmentInput {
    type Message = GeneratedTensorEnvironmentInputMessage;
    type Properties = EnvironmentInputProps;

    fn create(properties: Self::Properties, link: ComponentLink<Self>) -> Self {
        Self {
            properties,
            link,
            name: String::default(),
            shape_string: String::default(),
            value_generation_strategy: None,
        }
    }

    fn update(&mut self, msg: Self::Message) -> ShouldRender {
        match msg {
            GeneratedTensorEnvironmentInputMessage::UpdateName(s) => self.name = s,
            GeneratedTensorEnvironmentInputMessage::UpdateShapeString(s) => self.shape_string = s,
            GeneratedTensorEnvironmentInputMessage::UpdateValueGenerationStrategy(s) => {
                self.value_generation_strategy = Some(s)
            }
        }

        if let Some(value) = self.get_value() {
            self.properties.value_updated_callback.emit(value);
        }

        true
    }

    fn change(&mut self, properties: Self::Properties) -> ShouldRender {
        self.properties = properties;
        true
    }

    fn view(&self) -> Html {
        html! {
            <div>
                // Name text box
                <label for={"name"}>{"Name"}</label>
                <input name={"name"} type={"text"} oninput=self.link.callback(
                    |event: InputData| GeneratedTensorEnvironmentInputMessage::UpdateName(event.value)) />

                // Shape text box
                <label for={"shape"}>{"Shape (e.g. () or (3,32,32))"}</label>
                <input name={"shape"} type={"text"}
                    oninput=self.link.callback(|event: InputData| {
                        GeneratedTensorEnvironmentInputMessage::UpdateShapeString(event.value)
                    })
                />

                // Value generation radio buttons
                <input type={"radio"}
                    id={format!("random-{}", self.properties.id)}
                    name={format!("values-{}", self.properties.id)}
                    checked={match self.value_generation_strategy {
                        Some(ValueGenerationStrategy::Random) => true,
                        _ => false,
                    }}
                    oninput=self.link.callback(|_|
                        GeneratedTensorEnvironmentInputMessage::UpdateValueGenerationStrategy(
                            ValueGenerationStrategy::Random
                        ))
                />
                <label for={format!("random-{}", self.properties.id)}>{"random"}</label>

                <input type={"radio"}
                    id={format!("zeros-{}", self.properties.id)}
                    name={format!("values-{}", self.properties.id)}
                    checked={match self.value_generation_strategy {
                        Some(ValueGenerationStrategy::Zeros) => true,
                        _ => false,
                    }}
                    oninput=self.link.callback(|_|
                        GeneratedTensorEnvironmentInputMessage::UpdateValueGenerationStrategy(
                            ValueGenerationStrategy::Zeros
                        ))
                />
                <label for={format!("zeros-{}", self.properties.id)}>{"zeros"}</label>

                <input type={"radio"}
                    id={format!("ones-{}", self.properties.id)}
                    name={format!("values-{}", self.properties.id)}
                    checked={match self.value_generation_strategy {
                        Some(ValueGenerationStrategy::Ones) => true,
                        _ => false,
                    }}
                    oninput=self.link.callback(|_|
                        GeneratedTensorEnvironmentInputMessage::UpdateValueGenerationStrategy(
                            ValueGenerationStrategy::Ones
                        ))
                />
                <label for={format!("ones-{}", self.properties.id)}>{"ones"}</label>

                <input
                    type={"checkbox"}
                    id={"valid"}
                    disabled={true}
                    checked={match self.get_value() { Some(_) => true, _ => false}}
                />
                <label for={"valid"}>{"valid?"}</label>
            </div>
        }
    }
}

#[wasm_bindgen(start)]
pub fn start_app() {
    wasm_logger::init(wasm_logger::Config::default());
    yew::start_app::<App>();
}
