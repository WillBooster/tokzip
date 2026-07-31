import { javascriptModule } from '../generated/javascript.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; the model is validated at registration.
registerLanguageModule(javascriptModule);

export { javascriptModule };
